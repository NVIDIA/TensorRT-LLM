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

import contextlib
import os
import sys
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

from cbts.coverage.collection import hooks, process_roles
from cbts.coverage.collection.import_watch import ImportCompletionWatcher
from cbts.coverage.collection.pystart import PyStartTracker, read_source_config

if TYPE_CHECKING:
    from cbts.coverage.collection.channel import ContextSubscriber

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

    # Resolved at module load, where an ImportError surfaces normally: some callers of
    # these names run from inside a monkey-patched mpi4py handshake function, where it
    # must not be mistaken for a failure of their own guarded logic -- see _log_taint.
    from cbts.coverage.collection.channel import (
        TAINT_ATTRIBUTION,
        TAINT_INCOMPLETE,
        TAINT_NOT_SUBSCRIBED,
        TAINT_PY_START_UNAVAILABLE,
        ContextSubscriber,
        safe_print,
    )

    _CONFIG = os.getenv("CBTS_COVERAGE_CONFIG")

    # Source roots bound collection; include patterns match repository-relative Python paths.
    def _read_config(path: str) -> tuple[list[str], str, Optional[str], list[str]]:
        try:
            cp = read_source_config(path)
        except (OSError, configparser.Error) as exc:
            safe_print(
                f"[cbts] CBTS_COVERAGE_CONFIG={path!r} could not be read: {exc}; coverage disabled"
            )
            return [], ".", None, []
        src = [ln.strip() for ln in cp.get("run", "source", fallback="").splitlines() if ln.strip()]
        if not src:
            safe_print(f"[cbts] {path!r} has no [run] source entries; coverage disabled")
        data_file = cp.get("run", "data_file", fallback="")
        data_dir = os.path.dirname(data_file) or "."
        repository_root = cp.get("run", "repository_root", fallback="").strip() or None
        include = [
            ln.strip() for ln in cp.get("run", "include", fallback="").splitlines() if ln.strip()
        ]
        return src, data_dir, repository_root, include

    _src, _data_dir, _repository_root, _include = _read_config(_CONFIG)
    # Named by whoever renders the rcfile (make_coveragerc.sh writes it into data_file too, but
    # that copy is a filename, not an interface).
    _stage = os.environ.get("CBTS_STAGE", "").strip() or "stage"

    try:
        _PERIODIC_SAVE_SECONDS = max(0.1, float(os.environ.get("CBTS_PERIODIC_SAVE_SECONDS", "5")))
    except ValueError:
        _PERIODIC_SAVE_SECONDS = 5.0
    _stop_event = threading.Event()
    # Set by _subscribe_to_context_channel below; _final_save closes it so a normal
    # process exit sends LEAVE instead of leaving the producer to infer a crash from a bare EOF.
    _subscriber: Optional["ContextSubscriber"] = None

    _tracker = PyStartTracker(
        _src, _data_dir, _stage, repository_root=_repository_root, include=_include
    )

    def _log_taint(nodeid: Optional[str], kind: str, reason: str) -> None:
        """Record and log a taint this process observed about itself.

        Channel-observed taints reach ``_tracker.note_taint`` through
        ``_ActiveHooks.record_channel_taints`` below and are already logged once, by the
        outer pytest process, before it ever calls into this process's hooks -- calling
        this helper there too would print every one of them twice. This is only for
        taints a process raises about its own state, which have no other log line.

        Never raises: reporting a problem must not itself become a new one that
        disturbs the product, so a failure here (e.g. a broken stderr) is swallowed
        rather than surfaced -- best-effort is all a diagnostic owes. The print and the
        taint are guarded separately: a broken stderr must not also cost the taint,
        which needs no stderr at all to land in the leaf database.
        """
        safe_print(
            f"[cbts] tainted coverage ({kind}): {reason} for {nodeid or '<whole stage>'} "
            "in <this process>"
        )
        with contextlib.suppress(Exception):
            _tracker.note_taint(None, nodeid, kind, reason)

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
            try:
                return _tracker.save()
            except Exception as e:
                # Same guard as _save_active: this is called from cbts_plugin's
                # pytest_sessionfinish, after every test has already run and reported,
                # so a save failure here must not turn a passing session into a
                # reported error.
                safe_print(f"[cbts] flush save failed in pid {os.getpid()}: {e!r}")
                return None

        def record_channel_taints(
            self, taints: Iterable[tuple[Optional[str], str, str, str]]
        ) -> None:
            if _stop_event.is_set():
                return
            for process_uid, nodeid, kind, reason in taints:
                _tracker.note_taint(process_uid, nodeid, kind, reason)

        def record_process_diagnostics(self, diagnostics: Mapping[str, Mapping[str, str]]) -> None:
            if _stop_event.is_set():
                return
            for process_uid, facts in diagnostics.items():
                _tracker.note_process_diagnostic(process_uid, facts)

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
            safe_print(f"[cbts] periodic save failed in pid {os.getpid()}: {e!r}")

    def _final_save() -> None:
        """Last save for this process; idempotent, so STOP, atexit, and on_stop can all call it.

        Closes the channel subscriber (if any) after saving: a subscriber that just lets the
        process exit sends no goodbye, so the producer can only infer a crash from the bare
        EOF and has to taint every test it announces afterward on this subscriber's behalf.
        Closing here sends LEAVE first, telling the producer this departure was expected and
        this process's coverage is already safely on disk.
        """
        if _stop_event.is_set():
            return
        _stop_event.set()
        try:
            _tracker.save()
            _tracker.stop()
        except Exception as e:
            safe_print(f"[cbts] final save failed in pid {os.getpid()}: {e!r}")
        if _subscriber is not None:
            _subscriber.close()

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

        def _subscriber_diagnostics() -> dict[str, str]:
            """Facts about this process worth knowing if it is later dropped and asked about.

            Sent once at subscribe time because nothing can be read from the process
            anymore once it has exited. Add a key here to have it reach the merged
            report's ``process_diagnostic`` table for every future confirmed death or
            ordinary leave -- no protocol or schema change needed; see
            ``ContextSubscriber.subscribe``.
            """
            facts = {"cmdline": " ".join(getattr(sys, "orig_argv", None) or sys.argv)}
            pytest_case = os.environ.get("PYTEST_CURRENT_TEST", "").strip()
            if pytest_case:
                # Set by pytest itself, inherited if this process was forked from inside a
                # real pytest's test body (e.g. an inner-pytest batch's own worker).
                facts["pytest_current_test"] = pytest_case
            return facts

        def _subscribe_to_context_channel() -> None:
            """Join the outer pytest's channel, landing this process on the current test.

            Called exactly once per process: immediately for anything whose address is
            already in its environment, or from `_install_worker_sync_hook` for an MPI
            pool worker, once mpi4py has supplied it -- see README.md's "What happens
            to PY_START" section. A no-op, not an error, if no address is available
            even then (e.g. no channel at all this session; the coordinator's own
            `TAINT_NO_CHANNEL` already covers that).

            ``on_stop=_final_save`` can fire synchronously inside ``subscribe()`` if the
            channel had already sent its final STOP, or the producer had already exited,
            by the time this process joined: ``ContextSubscriber.subscribe`` reads the
            first frame inline, and either case delivers one immediately. Callers must
            re-check ``_stop_event`` after this returns before doing anything that assumes
            the tracker is still live.

            Never raises: coverage must never decide whether the product can start, and
            this can run inside a monkey-patched mpi4py handshake function -- any failure,
            anticipated or not, is reported and self-tainted rather than propagated.
            """
            global _subscriber
            try:
                _subscriber = ContextSubscriber.subscribe(
                    on_context=_active_hooks.switch_test_context,
                    on_stop=_final_save,
                    on_flush=_save_active,
                    identity=_tracker.process_uid,
                    diagnostics=_subscriber_diagnostics(),
                )
            except Exception as exc:
                # This runs at module level for most roles (see the call site below) --
                # nothing above catches a failure here, so even the report itself must
                # not be able to raise.
                safe_print(
                    f"[cbts] context subscribe failed in pid {os.getpid()}: {exc}: "
                    f"{exc.__cause__!r}"
                )
                # Nothing else can flag this: never having joined, this process is
                # unknown to the producer. Two separate doubts follow. Everything it
                # records lands on the context it was spawned with, so those rows may
                # belong to later tests it silently served. And every other test in the
                # stage is missing its coverage -- which tests, it cannot say, having
                # heard no announcement, so that half is stage-scoped. A reachable
                # address that failed to connect is not retried: that is a real
                # failure, not a not-yet-announced one.
                _log_taint(_tracker._ctx, TAINT_ATTRIBUTION, TAINT_NOT_SUBSCRIBED)
                _log_taint("", TAINT_INCOMPLETE, TAINT_NOT_SUBSCRIBED)

        def _activate_tracker() -> None:
            """Subscribe to the context channel if possible, then enable PY_START.

            Called once activation is worth attempting: immediately for a process whose
            address is already in its environment, or from `_install_worker_sync_hook`
            for an MPI pool worker, once mpi4py has supplied it. ``_tracker.start()``
            is guarded the same way `_subscribe_to_context_channel` guards itself: a
            failure here must become a taint, never an exception the real mpi4py
            handshake this can run inside of would have to deal with.
            """
            _subscribe_to_context_channel()
            if _stop_event.is_set():
                # subscribe() already ran _final_save synchronously (see its
                # docstring): this process's coverage is saved and done. Starting
                # PY_START now would only re-enable it for the rest of the process's
                # lifetime with every future save silently no-op'd on _stop_event --
                # pure overhead, none of it ever written.
                return
            try:
                started = _tracker.start()
            except Exception:
                started = False
            if not started:
                # sys.monitoring's PY_START tool could not be enabled (see
                # PyStartTracker.start): this process will record zero touches for its
                # entire life. Nothing else can see that to flag it -- it must report
                # on itself, stage-scoped, since it may go on to serve many tests, all
                # silently blank.
                _log_taint("", TAINT_INCOMPLETE, TAINT_PY_START_UNAVAILABLE)

        def _install_pool_patch() -> None:
            from cbts.coverage.collection.pool import install_pool_env_patch

            install_pool_env_patch()

        def _install_worker_sync_hook() -> None:
            """Patch mpi4py so an MPI pool worker activates once its coordinator's env lands.

            ``_lib._sync_set_data`` is what a spawned ``MPIPoolExecutor`` worker calls
            during its MPI sync handshake to apply the coordinator's env payload
            (``os.environ.update(data.pop('env'))``) -- the one point this process is
            guaranteed to have the channel address. It also runs before ``server_exec``
            (the task loop, the only place product code runs), so activating here loses
            nothing compared to activating at interpreter startup. See README.md's "What
            happens to PY_START" section for the full story.
            """
            from mpi4py.futures import _lib

            marker = "_cbts_patched_sync_data"
            if getattr(_lib._sync_set_data, marker, False):
                return
            original = _lib._sync_set_data

            def _patched(data):
                result = original(data)
                try:
                    _activate_tracker()
                except Exception:
                    # Last-resort net: _activate_tracker already guards its own steps,
                    # so this should never fire, but this function stands in for a real
                    # mpi4py handshake step, and coverage must never be able to corrupt
                    # or abort that for a failure nothing anticipated.
                    _log_taint("", TAINT_INCOMPLETE, TAINT_PY_START_UNAVAILABLE)
                return result

            setattr(_patched, marker, True)
            _lib._sync_set_data = _patched

        # The pool env patch goes on as mpi4py.futures lands, before any MPIPoolExecutor can be
        # constructed. Both pytest roles are covered already -- the outer one by the plugin's
        # pytest_configure, the inner one above -- and a process that never builds a pool just
        # leaves the patch unused. It's usually the *coordinator* (not a worker) that constructs
        # the pool.
        _watched_imports = {"mpi4py.futures": _install_pool_patch}
        if process_roles.is_mpi_pool_worker():
            # This process's channel address arrives only through mpi4py's own sync
            # handshake, not at exec like an ordinary child's -- see
            # process_roles.is_mpi_pool_worker and _install_worker_sync_hook. Defer
            # activation to that hook instead of attempting it now, when it is
            # guaranteed to find no address yet.
            _watched_imports["mpi4py.futures._lib"] = _install_worker_sync_hook
            sys.meta_path.insert(0, ImportCompletionWatcher(_watched_imports))
        else:
            sys.meta_path.insert(0, ImportCompletionWatcher(_watched_imports))
            # No barrier to protect for any other product process (trtllm-serve, disagg
            # helpers, an mpirun-launched MPICommExecutor sibling, ...): activate
            # immediately, same as a pytest process, so its import phase gets recorded too.
            _activate_tracker()

    if _ROLE == _INNER_PYTEST:
        # The inner pytest runs without -p cbts_plugin, so nothing else will install the pool
        # env patch for it; do it here rather than through the watcher thread below.
        try:
            from cbts.coverage.collection.pool import install_pool_env_patch

            install_pool_env_patch()
        except Exception as _exc:
            # Module-level, like the block above: a report failure here must not
            # raise either, or it takes this whole sitecustomize import down with it.
            safe_print(f"[cbts] inner-pytest mpi patch skipped in pid {os.getpid()}: {_exc!r}")

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
