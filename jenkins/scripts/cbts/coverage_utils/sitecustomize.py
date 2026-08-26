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


# Interpreter options that consume the token after them, so the scan below can tell a script
# path from an option argument (``python -X importtime -m pytest`` must still yield "pytest").
_OPTIONS_WITH_ARGUMENT = frozenset(("-W", "-X", "--check-hash-based-pycs"))


def _launch_target():
    """How this interpreter was started, as ``(module, script_basename)``.

    Exactly one is set: ``module`` for ``-m pkg.mod``, ``script_basename`` for a
    script or console-script path. Both are ``None`` for ``-c`` and for a bare
    REPL. Parsing stops at the launch target, so a program's own arguments can
    never be mistaken for the interpreter's.
    """
    argv = list(getattr(sys, "orig_argv", sys.argv) or [""])[1:]
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-m":
            return (argv[index + 1] if index + 1 < len(argv) else None), None
        if token == "-c":
            return None, None
        if token in _OPTIONS_WITH_ARGUMENT:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return None, os.path.basename(token)
    return None, None


def _is_dependency_build_process():
    """Return True for pip / setuptools / native build-tool processes, which opt the subtree out.

    These are spawned by pip and the PEP 517 backend rather than by our own code, so there is
    nobody to hand them an explicit role; the launch target is the available signal.
    """
    module, script = _launch_target()
    if module is not None:
        return module.split(".", 1)[0] in {"pip", "build", "pyproject_hooks", "setuptools"}
    if script is None:
        return False
    script = script.lower()
    return script in {
        "pip",
        "pip3",
        "cmake",
        "ninja",
        "ninja-build",
        "meson",
        "setup.py",
        "_in_process.py",
    }


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

    raylet spawns these, not our own code, so there is nobody to hand them an explicit role;
    the launch target is the available signal.
    """
    module, script = _launch_target()
    if module is not None:
        # Dotted-prefix match over the module namespace, so e.g.
        # ray.autoscaler._private.monitor is covered without matching a path that
        # merely contains the text.
        return module == "ray" or module.startswith(
            ("ray.autoscaler.", "ray.dashboard.", "ray._private.")
        )
    return script in {"default_worker.py", "setup_worker.py"}


# What this process is, told to us by whoever launched it: "outer_pytest" (the stage's pytest,
# set by jenkins/L0_Test.groovy) or "inner_pytest" (the unit-test batch, set by
# tests/integration/defs/test_unittests.py). Anything else -- MPI pool workers, trtllm-serve,
# helper subprocesses -- runs unlabelled and is treated as a product process.
#
# Popped rather than read: a role describes one process, while everything else CBTS puts in the
# environment (CBTS_COVERAGE_CONFIG, CBTS_TEST_ID, ...) is meant to be inherited. Consuming it
# here, before this process can spawn anything, keeps a child from answering to its parent's role.
_ROLE = os.environ.pop("CBTS_PROCESS_ROLE", "").strip()
_OUTER_PYTEST = "outer_pytest"
_INNER_PYTEST = "inner_pytest"
_IS_PYTEST_PROCESS = _ROLE in (_OUTER_PYTEST, _INNER_PYTEST)

# Drop the gate var so build tooling / Ray infra and everything they spawn opt out of instrumentation.
if os.getenv("CBTS_COVERAGE_CONFIG") and (
    _is_dependency_build_process() or _is_ray_infra_process()
):
    os.environ.pop("CBTS_COVERAGE_CONFIG", None)


if os.getenv("CBTS_COVERAGE_CONFIG"):
    import atexit
    import configparser
    import importlib.abc
    import importlib.util
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
            return [], "."
        src = [ln.strip() for ln in cp.get("run", "source", fallback="").splitlines() if ln.strip()]
        data_file = cp.get("run", "data_file", fallback="")
        data_dir = os.path.dirname(data_file) or "."
        return src, data_dir

    _src, _data_dir = _read_config(_CONFIG)
    # Named by whoever renders the rcfile (make_coveragerc.sh writes it into data_file too, but
    # that copy is a filename, not an interface).
    _stage = os.environ.get("CBTS_STAGE", "").strip() or "stage"

    try:
        _PERIODIC_SAVE_SECONDS = max(0.1, float(os.environ.get("CBTS_PERIODIC_SAVE_SECONDS", "5")))
    except ValueError:
        _PERIODIC_SAVE_SECONDS = 5.0
    _stop_event = threading.Event()

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
        if _stop_event.is_set():
            return
        try:
            _tracker.save()
        except Exception as e:
            print(f"[cbts] periodic save failed in pid {os.getpid()}: {e!r}", file=sys.stderr)

    def _final_save():
        """Last save for this process; idempotent, so STOP and atexit can both call it."""
        if _stop_event.is_set():
            return
        _stop_event.set()
        try:
            _tracker.save()
            _tracker.stop()
        except Exception as e:
            print(f"[cbts] final save failed in pid {os.getpid()}: {e!r}", file=sys.stderr)

    atexit.register(_final_save)

    # Subprocesses inherit the current nodeid via CBTS_TEST_ID; the outer pytest re-switches per test via the plugin.
    _initial_nodeid = os.environ.get("CBTS_TEST_ID", "").strip()
    if _initial_nodeid:
        switch_test_context(_initial_nodeid)

    if _IS_PYTEST_PROCESS:
        # A pytest process is already running product-independent code worth recording (collection,
        # fixtures), and an inner pytest carries a real CBTS_TEST_ID from the start, so its import
        # phase belongs to that entry.
        _tracker.start()
    else:
        # Product processes enable PY_START only once the framework's first import has finished (or
        # CBTS_WORKER_ACTIVATE_MAX_SECONDS elapses). Instrumenting that cold import is slow enough to
        # time out an MpiPoolSession wait_shutdown identity barrier. Module and class bodies executed
        # before activation go unrecorded here; functions a test exercises are re-entered afterwards.
        try:
            _activate_max = max(
                1.0, float(os.environ.get("CBTS_WORKER_ACTIVATE_MAX_SECONDS", "120"))
            )
        except ValueError:
            _activate_max = 120.0
        # Product top-level import names taken from the coverage source roots (e.g. "tensorrt_llm").
        _product_tops = {os.path.basename(p.rstrip("/")) for p in _src if p}
        _framework_ready = threading.Event()
        _activation_lock = threading.Lock()

        def _subscribe_to_context_channel():
            """Join the outer pytest's channel, landing this process on the current test.

            Deferred to here rather than run at bootstrap: an mpi4py pool worker only
            learns the channel address from the env payload its sync handshake applies,
            which is after interpreter startup but before it runs any task.
            """
            try:
                from cbts_channel import ContextSubscriber

                ContextSubscriber.subscribe(
                    on_context=switch_test_context,
                    on_stop=_final_save,
                )
            except Exception as exc:
                print(
                    f"[cbts] context subscribe failed in pid {os.getpid()}: {exc!r}",
                    file=sys.stderr,
                )

        def _activate_tracker():
            """Enable PY_START once, from whichever of the two paths below arrives first."""
            with _activation_lock:
                if _framework_ready.is_set() or _stop_event.is_set():
                    return
                _framework_ready.set()
                # Subscribe first: recording must not start on a stale context.
                _subscribe_to_context_channel()
                _tracker.start()

        _resolving = threading.local()

        class _ImportCompletionWatcher(importlib.abc.MetaPathFinder):
            """Meta-path finder that runs a hook once a watched module finishes executing.

            Hands resolution back to ``importlib.util.find_spec`` and swaps the resolved
            loader for one that calls back after ``exec_module`` returns, so a hook keys off
            the actual end of the module body rather than sampling import state. Running on
            the importing thread leaves no window between a module becoming usable and its
            hook having run.

            ``callbacks`` maps module name to hook. Entries are dropped as they fire, so each
            hook runs once and every later import costs one dict miss; the finder stays in
            ``sys.meta_path`` rather than removing itself, which would mutate that list
            mid-import while another thread may be walking it.
            """

            def __init__(self, callbacks):
                self._callbacks = callbacks

            def find_spec(self, fullname, path=None, target=None):
                # The delegated lookup walks sys.meta_path again and arrives back here; the
                # thread-local flag makes that second pass decline instead of recursing.
                if fullname not in self._callbacks or getattr(_resolving, "active", False):
                    return None
                _resolving.active = True
                try:
                    spec = importlib.util.find_spec(fullname)
                except (ImportError, ValueError):
                    return None
                finally:
                    _resolving.active = False
                if spec is None or not hasattr(spec.loader, "exec_module"):
                    return spec
                spec.loader = _WatchedLoader(spec.loader, fullname, self.fire)
                return spec

            def fire(self, fullname):
                callback = self._callbacks.pop(fullname, None)
                if callback is None:
                    return
                try:
                    callback()
                except Exception as exc:
                    print(
                        f"[cbts] import hook for {fullname} failed in pid {os.getpid()}: {exc!r}",
                        file=sys.stderr,
                    )

        class _WatchedLoader(importlib.abc.Loader):
            """Loader proxy that calls back once the wrapped ``exec_module`` returns."""

            def __init__(self, inner, fullname, fire):
                self._inner = inner
                self._fullname = fullname
                self._fire = fire

            def create_module(self, spec):
                return self._inner.create_module(spec)

            def exec_module(self, module):
                try:
                    self._inner.exec_module(module)
                finally:
                    # fire() swallows hook failures; an import must not break over coverage.
                    self._fire(self._fullname)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def _install_pool_patch():
            from cbts_pool import install_expected_workers_patch

            install_expected_workers_patch()

        # The pool accounting patch goes on as mpi4py.futures lands, before any MPIPoolExecutor
        # can be constructed. Both pytest roles are covered already -- the outer one by the
        # plugin's pytest_configure, the inner one above -- and a process that never builds a
        # pool just leaves the patch unused.
        _watched_imports = dict.fromkeys(_product_tops, _activate_tracker)
        _watched_imports["mpi4py.futures"] = _install_pool_patch
        sys.meta_path.insert(0, _ImportCompletionWatcher(_watched_imports))

        def _deferred_activate():
            # Backstop for a process that never imports the framework; no-ops once the
            # watcher above has already activated.
            _framework_ready.wait(_activate_max)
            _activate_tracker()

        threading.Thread(
            target=_deferred_activate,
            daemon=True,
            name="cbts-deferred-activate",
        ).start()

    if _ROLE == _INNER_PYTEST:
        # The inner pytest runs without -p cbts_plugin, so nothing else will install the pool
        # accounting patch for it; do it here rather than through the watcher thread below.
        try:
            from cbts_pool import install_expected_workers_patch

            install_expected_workers_patch()
        except Exception as _exc:
            print(
                f"[cbts] inner-pytest mpi patch skipped in pid {os.getpid()}: {_exc!r}",
                file=sys.stderr,
            )

    # Every instrumented process saves periodically so its coverage survives a non-clean exit;
    # pool workers in particular lose their atexit save when the pool is torn down at test end.
    def _periodic_save():
        while not _stop_event.wait(_PERIODIC_SAVE_SECONDS):
            _save_active()

    threading.Thread(
        target=_periodic_save,
        daemon=True,
        name="cbts-periodic-save",
    ).start()
