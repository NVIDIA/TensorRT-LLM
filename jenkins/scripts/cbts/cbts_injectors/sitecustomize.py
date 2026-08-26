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
"""Per-process function-level coverage bootstrap for CBTS Layer C.

Active only when CBTS_COVERAGE_CONFIG is set. Loaded automatically by Python's ``site``
machinery in every subprocess of an instrumented stage: this file's directory
(``cbts_injectors/``) is put on ``PYTHONPATH`` for exactly that reason, alongside the outer
``cbts`` dir that lets it reach ``cbts.coverage.collection.*`` as a normal package.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

from cbts.coverage.collection import hooks, process_roles
from cbts.coverage.collection.import_watch import ImportCompletionWatcher
from cbts.coverage.collection.pystart import PyStartTracker

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
    process_roles.is_dependency_build_process() or process_roles.is_ray_infra_process()
):
    os.environ.pop("CBTS_COVERAGE_CONFIG", None)


if os.getenv("CBTS_COVERAGE_CONFIG"):
    import atexit
    import configparser
    import threading

    _CONFIG = os.getenv("CBTS_COVERAGE_CONFIG")

    # Read [run] source (product roots) and data_file (dir + stage name) from the rendered rcfile.
    def _read_config(path: str) -> tuple[list[str], str]:
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

    class _ActiveHooks(hooks.Hooks):
        """Real implementations, installed as ``hooks.active`` for the rest of this process.

        Each overrides its base-class contract identically apart from what it forwards to
        ``_tracker``; see ``cbts.coverage.collection.hooks.Hooks`` for what each one means.
        Every method no-ops once ``_stop_event`` is set, rather than recording (or worse,
        enabling PY_START to record) into a tracker that has already taken its final save.
        """

        def switch_test_context(self, nodeid: Optional[str]) -> None:
            if _stop_event.is_set():
                return
            _tracker.switch_test_context(nodeid or "")

        def record_test_outcome(self, nodeid: Optional[str], outcome: str) -> None:
            if _stop_event.is_set():
                return
            _tracker.record_outcome(nodeid or "", outcome)

        def flush_coverage(self) -> Optional[str]:
            if _stop_event.is_set():
                return None
            return _tracker.save()

        def record_channel_taints(
            self, taints: Iterable[tuple[Optional[str], str, str, str]]
        ) -> None:
            if _stop_event.is_set():
                return
            for process_uid, nodeid, kind, reason in taints:
                _tracker.note_taint(process_uid, nodeid, kind, reason)

        def note_expected_workers(self, nodeid: Optional[str], n: int) -> None:
            if _stop_event.is_set():
                return
            _tracker.note_expected_workers(nodeid or "", n)

    # Installed so peers (the pool patch, the pytest plugin) can call back without importing
    # this guest-injected module by name; see cbts.coverage.collection.hooks.
    _active_hooks = _ActiveHooks()
    hooks.active = _active_hooks

    def _save_active() -> None:
        if _stop_event.is_set():
            return
        try:
            _tracker.save()
        except Exception as e:
            print(f"[cbts] periodic save failed in pid {os.getpid()}: {e!r}", file=sys.stderr)

    def _final_save() -> None:
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
        _active_hooks.switch_test_context(_initial_nodeid)

    if _IS_PYTEST_PROCESS:
        # A pytest process is already running product-independent code worth recording (collection,
        # fixtures), and an inner pytest carries a real CBTS_TEST_ID from the start, so its import
        # phase belongs to that entry.
        _tracker.start()
    else:
        _framework_ready = threading.Event()
        _activation_lock = threading.Lock()

        def _subscribe_to_context_channel() -> None:
            """Join the outer pytest's channel, landing this process on the current test.

            ``on_stop=_final_save`` can fire synchronously inside ``subscribe()`` if the
            channel had already sent its final STOP, or the producer had already exited,
            by the time this process joined: ``ContextSubscriber.subscribe`` reads the
            first frame inline, and either case delivers one immediately. Callers must
            re-check ``_stop_event`` after this returns before doing anything that assumes
            the tracker is still live.
            """
            from cbts.coverage.collection.channel import (
                TAINT_ATTRIBUTION,
                TAINT_INCOMPLETE,
                TAINT_NOT_SUBSCRIBED,
                ContextSubscriber,
            )

            try:
                ContextSubscriber.subscribe(
                    on_context=_active_hooks.switch_test_context,
                    on_stop=_final_save,
                    identity=_tracker.process_uid,
                )
            except Exception as exc:
                # Reported rather than raised: this runs inside the framework's import,
                # and coverage must not decide whether the product can start.
                print(
                    f"[cbts] context subscribe failed in pid {os.getpid()}: {exc}: "
                    f"{exc.__cause__!r}",
                    file=sys.stderr,
                )
                # Nothing else can flag this: never having joined, this process is
                # unknown to the producer. Two separate doubts follow. Everything it
                # records lands on the context it was spawned with, so those rows may
                # belong to later tests it silently served. And every other test in the
                # stage is missing its coverage -- which tests, it cannot say, having
                # heard no announcement, so that half is stage-scoped.
                _tracker.note_taint(None, _tracker._ctx, TAINT_ATTRIBUTION, TAINT_NOT_SUBSCRIBED)
                _tracker.note_taint(None, "", TAINT_INCOMPLETE, TAINT_NOT_SUBSCRIBED)

        def _activate_tracker(timed_out: bool = False) -> None:
            """Enable PY_START, reachable from either the import watcher or the backstop below.

            ``timed_out`` is set only by the backstop, meaning the framework import watcher
            never fired within the deadline. That taints the process: nothing was recorded yet
            (PY_START isn't on until this call), so there is no attribution risk, only an
            incomplete one for whatever ran during the delay.
            """
            with _activation_lock:
                if _framework_ready.is_set() or _stop_event.is_set():
                    return
                _framework_ready.set()
                # Subscribe first: recording must not start on a stale context.
                _subscribe_to_context_channel()
                if _stop_event.is_set():
                    # subscribe() already ran _final_save synchronously (see its
                    # docstring): this process's coverage is saved and done. Starting
                    # PY_START now would only re-enable it for the rest of the process's
                    # lifetime with every future save silently no-op'd on _stop_event --
                    # pure overhead, none of it ever written.
                    return
                if timed_out:
                    from cbts.coverage.collection.channel import (
                        TAINT_ACTIVATION_TIMEOUT,
                        TAINT_INCOMPLETE,
                    )

                    _tracker.note_taint(None, "", TAINT_INCOMPLETE, TAINT_ACTIVATION_TIMEOUT)
                _tracker.start()

        def _install_pool_patch() -> None:
            from cbts.coverage.collection.pool import install_expected_workers_patch

            install_expected_workers_patch()

        # The pool accounting patch goes on as mpi4py.futures lands, before any MPIPoolExecutor
        # can be constructed. Both pytest roles are covered already -- the outer one by the
        # plugin's pytest_configure, the inner one above -- and a process that never builds a
        # pool just leaves the patch unused. Every product process needs this watcher, whether
        # or not it defers its own activation below: it's usually the *coordinator* (not a
        # worker) that constructs the pool.
        _watched_imports = {"mpi4py.futures": _install_pool_patch}

        if process_roles.is_mpi_pool_worker():
            # Only an actual MPI pool worker risks tripping MpiPoolSession's wait_shutdown
            # identity barrier (tensorrt_llm/llmapi/mpi_session.py): that barrier's deadline
            # is shared with process spawn + this process's first framework import, so an
            # instrumented cold import can eat into (or blow) it. Defer PY_START until that
            # import finishes (or CBTS_WORKER_ACTIVATE_MAX_SECONDS elapses) so the barrier
            # sees a normal-speed worker; module and class bodies executed before activation
            # go unrecorded here, but functions a test exercises are re-entered afterwards.
            try:
                _activate_max = max(
                    1.0, float(os.environ.get("CBTS_WORKER_ACTIVATE_MAX_SECONDS", "120"))
                )
            except ValueError:
                _activate_max = 120.0
            # Product top-level import names taken from the coverage source roots (e.g. "tensorrt_llm").
            _product_tops = {str(os.path.basename(p.rstrip("/"))) for p in _src if p}
            _watched_imports.update(dict.fromkeys(_product_tops, _activate_tracker))
            sys.meta_path.insert(0, ImportCompletionWatcher(_watched_imports))

            def _deferred_activate() -> None:
                # Backstop for a worker that never imports the framework, or whose import is
                # abnormally slow. No-ops (via _activate_tracker's own guard) if the watcher
                # above already activated before the deadline.
                if not _framework_ready.wait(_activate_max):
                    _activate_tracker(timed_out=True)

            threading.Thread(
                target=_deferred_activate,
                daemon=True,
                name="cbts-deferred-activate",
            ).start()
        else:
            # No barrier to protect for any other product process (trtllm-serve, disagg
            # helpers, ...): activate immediately, same as a pytest process, so its import
            # phase gets recorded too.
            sys.meta_path.insert(0, ImportCompletionWatcher(_watched_imports))
            _activate_tracker()

    if _ROLE == _INNER_PYTEST:
        # The inner pytest runs without -p cbts_plugin, so nothing else will install the pool
        # accounting patch for it; do it here rather than through the watcher thread below.
        try:
            from cbts.coverage.collection.pool import install_expected_workers_patch

            install_expected_workers_patch()
        except Exception as _exc:
            print(
                f"[cbts] inner-pytest mpi patch skipped in pid {os.getpid()}: {_exc!r}",
                file=sys.stderr,
            )

    # Every instrumented process saves periodically so its coverage survives a non-clean exit;
    # pool workers in particular lose their atexit save when the pool is torn down at test end.
    def _periodic_save() -> None:
        while not _stop_event.wait(_PERIODIC_SAVE_SECONDS):
            _save_active()

    threading.Thread(
        target=_periodic_save,
        daemon=True,
        name="cbts-periodic-save",
    ).start()
