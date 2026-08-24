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


def _is_ray_infra_process():
    """Return True for Ray infrastructure / worker processes, which opt the subtree out.

    The Ray stage (TLLM_DISABLE_MPI=1) nests Ray under mpi4py pool workers: each pool worker
    calls ``ray.init(address="local")`` which spawns ``raylet``, ``gcs_server``, dashboard,
    log_monitor, autoscaler.monitor, runtime_env.agent, and pre-starts up to 224 ``default_worker.py``
    processes. All of them inherit ``CBTS_COVERAGE_CONFIG``/``PYTHONPATH`` via default env inheritance.

    Activating CBTS in ``default_worker.py`` adds enough Python startup / sys.monitoring PY_START
    overhead that the workers can't register with raylet before its ``worker_pool.cc:600`` timeout,
    so raylet keeps spawning more, and the driver's ``ray.init()`` hangs in ``RegisterClient`` forever
    (observed in test_disaggregated_* under the Ray orchestrator stage). Opt out here so Ray's
    hot spawn path stays fast; the mpi4py pool worker still records coverage for the LLM API surface,
    and the RayGPUWorker actor itself lives inside a ``default_worker.py`` so it's uninstrumented too.
    """
    argv = getattr(sys, "orig_argv", sys.argv) or [""]
    # Match on filename / module tokens Ray uses when it spawns Python children.
    indicators = (
        "default_worker.py",
        "setup_worker.py",
        "ray.autoscaler",
        "ray.dashboard",
        "ray._private.log_monitor",
        "ray._private.runtime_env.agent",
        "ray._private.workers",
        "/ray/dashboard/",
        "/ray/autoscaler/",
    )
    joined = " ".join(argv)
    return any(ind in joined for ind in indicators)


# Drop the gate var so build tooling / Ray infra and everything they spawn opt out of instrumentation.
if os.getenv("CBTS_COVERAGE_CONFIG") and (
    _is_dependency_build_process() or _is_ray_infra_process()
):
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

    try:
        _PERIODIC_SAVE_SECONDS = max(0.1, float(os.environ.get("CBTS_PERIODIC_SAVE_SECONDS", "5")))
    except ValueError:
        _PERIODIC_SAVE_SECONDS = 5.0
    _stop_event = threading.Event()

    # While this file exists (created by result collection) no process saves.
    _STOP_FILE = os.environ.get("CBTS_STOP_FILE", "")

    def _frozen():
        return bool(_STOP_FILE) and os.path.exists(_STOP_FILE)

    _tracker = PyStartTracker(_src, _data_dir, _stage)

    def switch_test_context(nodeid):
        """Switch the active test context; each test's entered functions are recorded separately."""
        if _stop_event.is_set():
            return
        _tracker.switch_test_context(nodeid or "")

    def record_test_outcome(nodeid, outcome):
        """Record a test's pytest outcome for the merge-side completeness signal (outer pytest only)."""
        if _stop_event.is_set():
            return
        _tracker.record_outcome(nodeid or "", outcome)

    def note_expected_workers(nodeid, n):
        """Record that the coordinator spawned n subprocess pool workers for a test."""
        if _stop_event.is_set():
            return
        _tracker.note_expected_workers(nodeid or "", n)

    def _save_active():
        if _frozen():
            return
        try:
            _tracker.save()
        except Exception as e:
            print(f"[cbts] periodic save failed in pid {os.getpid()}: {e!r}", file=sys.stderr)

    def _final_save():
        _stop_event.set()
        try:
            if not _frozen():
                _tracker.save()
            _tracker.stop()
        except Exception as e:
            print(f"[cbts] final save failed in pid {os.getpid()}: {e!r}", file=sys.stderr)

    atexit.register(_final_save)

    # sys.orig_argv preserves the launching cmdline; sys.argv has not yet gained "pytest" when sitecustomize runs.
    _orig_argv = getattr(sys, "orig_argv", sys.argv)
    _is_pytest_main = any("pytest" in a for a in _orig_argv[:4])
    _is_nested_pytest = _parent_is_pytest() and _is_pytest_main
    # mpi4py.futures pool workers don't spawn pools (so they skip the mpi patcher) but still run the
    # periodic saver and the marker poller so their per-test coverage is saved and attributed.
    _is_mpi_pool_worker = any("mpi4py.futures" in a for a in _orig_argv)
    _skip_daemons = _is_pytest_main or _is_mpi_pool_worker

    # Subprocesses inherit the current nodeid via CBTS_TEST_ID; the outer pytest re-switches per test via the plugin.
    _initial_nodeid = os.environ.get("CBTS_TEST_ID", "").strip()
    if _initial_nodeid:
        switch_test_context(_initial_nodeid)

    # mpi4py.futures pool workers enable PY_START only after the product framework's first import
    # settles (or CBTS_WORKER_ACTIVATE_MAX_SECONDS); every other process enables it now. Deferring
    # keeps a wait_shutdown MpiPoolSession identity barrier from timing out on the instrumented cold
    # import; coverage is unaffected since functions a test exercises are re-entered after activation.
    _defer_worker_activation = _is_mpi_pool_worker and os.environ.get(
        "CBTS_DEFER_WORKER_ACTIVATION", "1"
    ) not in ("0", "false", "False", "")

    if not _defer_worker_activation:
        _tracker.start()
    else:
        try:
            _activate_max = max(
                1.0, float(os.environ.get("CBTS_WORKER_ACTIVATE_MAX_SECONDS", "120"))
            )
        except ValueError:
            _activate_max = 120.0
        # Product top-level import names taken from the coverage source roots (e.g. "tensorrt_llm").
        _product_tops = {os.path.basename(p.rstrip("/")) for p in _src if p}

        def _framework_imported():
            for _name in _product_tops:
                _mod = sys.modules.get(_name)
                _spec = getattr(_mod, "__spec__", None) if _mod is not None else None
                if _mod is not None and not getattr(_spec, "_initializing", False):
                    return True
            return False

        def _deferred_activate():
            _waited = 0.0
            _step = 0.2
            while not _stop_event.is_set() and _waited < _activate_max:
                if _framework_imported():
                    break
                _stop_event.wait(_step)
                _waited += _step
            if not _stop_event.is_set():
                _tracker.start()

        threading.Thread(
            target=_deferred_activate,
            daemon=True,
            name="cbts-deferred-activate",
        ).start()

    if _is_nested_pytest:
        # Inner pytest: install the pool accounting/env patch synchronously instead of via the watcher.
        try:
            from cbts_plugin import install_expected_workers_patch

            install_expected_workers_patch()
        except Exception as _exc:
            print(
                f"[cbts] nested-pytest mpi patch skipped in pid {os.getpid()}: {_exc!r}",
                file=sys.stderr,
            )

    # Every instrumented process saves periodically so its coverage survives a non-clean exit;
    # pool workers in particular lose their atexit save when the pool is torn down at test end.
    def _periodic_save():
        while not _stop_event.wait(_PERIODIC_SAVE_SECONDS):
            if _frozen():
                _stop_event.set()
                return
            _save_active()

    threading.Thread(
        target=_periodic_save,
        daemon=True,
        name="cbts-periodic-save",
    ).start()

    if not _skip_daemons:
        # Coordinator / long-lived non-pytest processes install the pool accounting/env patch before
        # they spawn a pool; pool workers and the outer pytest don't spawn pools, so they skip this.

        def _watch_mpi_pool():
            # Wait until mpi4py.futures is imported so installing the patch triggers no racing import.
            while not _stop_event.is_set():
                if "mpi4py.futures" in sys.modules:
                    try:
                        from cbts_plugin import install_expected_workers_patch

                        install_expected_workers_patch()
                    except Exception as exc:
                        print(
                            f"[cbts] pool patch in pid {os.getpid()} failed: {exc!r}",
                            file=sys.stderr,
                        )
                    return
                _stop_event.wait(0.1)

        threading.Thread(
            target=_watch_mpi_pool,
            daemon=True,
            name="cbts-pool-patcher",
        ).start()

    # Pool workers and long-lived non-pytest processes follow the per-test marker file to switch
    # context; a pool worker would otherwise record every test it serves under one inherited (often
    # empty) context. The outer pytest switches context via the plugin instead.
    if not _is_pytest_main:
        from cbts_plugin import DEFAULT_MARKER_FILE

        _MARKER_FILE = os.environ.get("CBTS_MARKER_FILE", DEFAULT_MARKER_FILE)

        def _poll_marker():
            last_seen = _initial_nodeid
            while not _stop_event.is_set():
                try:
                    with open(_MARKER_FILE) as f:
                        nodeid = f.read().strip()
                    if nodeid and nodeid != last_seen:
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
