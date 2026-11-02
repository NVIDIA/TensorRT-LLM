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
"""Pytest plugin for CBTS Layer C per-test coverage attribution.

Loaded only by a pytest process (``-p cbts_plugin``), so pytest is imported
unconditionally and the hooks below are always bound: whatever imports this
module gets the same plugin. Anything a non-pytest process needs lives in
``cbts.coverage.collection.pool``.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterable, Optional

import pytest
from cbts.coverage.collection import hooks
from cbts.coverage.collection.channel import (
    ADDRESS_ENV,
    TAINT_ATTRIBUTION,
    TAINT_CONTEXT_SWITCH_FAILED,
    TAINT_INCOMPLETE,
    TAINT_NO_CHANNEL,
    TAINT_SESSION_TEARDOWN_FAILED,
    ContextServer,
    safe_print,
)
from cbts.coverage.collection.pool import install_pool_env_patch

_server: Optional[ContextServer] = None


def pytest_configure(config: pytest.Config) -> None:  # noqa: D401 - pytest hook
    """Install the pool env patch and open the context channel.

    Coverage must never decide whether the test session can start. ``install_pool_env_patch``
    already guards and self-taints its own failures; opening the channel is best-effort here,
    reported rather than raised, and its failure is folded into a taint at session end below
    (``_server`` stays ``None``, so ``pytest_sessionfinish``'s own ``TAINT_NO_CHANNEL`` path
    already covers it).
    """
    del config
    global _server
    install_pool_env_patch()
    try:
        _server = ContextServer.start()
    except Exception as exc:
        # Report only: CBTS setup failure doesn't block test run. Nothing wraps this
        # hook, so the report itself must not be able to raise either -- that would
        # abort the whole test session before anything runs.
        cause = getattr(exc, "__cause__", None)
        safe_print(f"[cbts] {exc}: {cause!r}; subprocesses keep their inherited CBTS_TEST_ID")
        return
    # Set before any test can spawn: ordinary children inherit it at exec, and the patched
    # MPIPoolExecutor reads it here to forward into its workers' env payload.
    os.environ[ADDRESS_ENV] = _server.address


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Optional[pytest.Item]):  # noqa: D401
    """Switch this process to the current test, then bring every subscriber with it.

    Everything before ``yield`` runs ahead of the real test; coverage must never be
    able to stop that test from running at all, so a failure here is reported and
    self-tainted rather than raised.
    """
    del nextitem
    nodeid = item.nodeid

    # The bootstrap default for a process that starts before it can subscribe.
    os.environ["CBTS_TEST_ID"] = nodeid

    try:
        hooks.active.switch_test_context(nodeid)
        if _server is not None and not _server.announce(nodeid):
            safe_print(
                f"[cbts] not all subscribers acknowledged {nodeid}; their next touches may "
                "land on the previous test"
            )
    except Exception as exc:
        safe_print(f"[cbts] context switch to {nodeid} failed: {exc!r}")
        # The current test's own rows may belong to the superseded context; whether
        # this is a one-off or the mechanism is now broken for the rest of the
        # session cannot be told from here, so every later test is tainted too.
        with contextlib.suppress(Exception):
            hooks.active.record_channel_taints(
                [
                    (None, nodeid, TAINT_ATTRIBUTION, TAINT_CONTEXT_SWITCH_FAILED),
                    (None, "", TAINT_INCOMPLETE, TAINT_CONTEXT_SWITCH_FAILED),
                ]
            )

    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):  # noqa: D401
    """Record each test's outcome so the merge can flag coverage that isn't safe to trust.

    ``outcome.get_result()`` has already captured the real result by the time this
    runs; reported rather than raised, so a recording failure cannot turn into a
    session-level hook error over a test that otherwise ran fine.
    """
    del call
    outcome = yield
    try:
        report = outcome.get_result()
        # The call phase is the test body; a non-passing setup or teardown is the
        # test's effective outcome (a failing teardown downgrades an already-recorded
        # pass).
        if report.when == "call" or (
            report.when in ("setup", "teardown") and report.outcome != "passed"
        ):
            hooks.active.record_test_outcome(item.nodeid, report.outcome)
    except Exception as exc:
        safe_print(f"[cbts] outcome recording for {item.nodeid} failed: {exc!r}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: D401
    """Close the channel and wait for every subscriber to save and leave.

    All tests have already run and reported by now, but exitstatus has not been
    acted on yet -- reported rather than raised here too, so a teardown failure in
    coverage bookkeeping cannot turn a fully passing session into a reported error.
    """
    del session, exitstatus
    global _server
    try:
        if _server is None:
            # Stage-scoped and both kinds: with no channel, every process spent the
            # session on its spawn context, so every test may hold another test's rows
            # and be missing its own. None (not ""): this is a taint about the pytest
            # coordinator's own leaf database, not an unidentified subscriber.
            _record_taints(
                [
                    (None, "", TAINT_ATTRIBUTION, TAINT_NO_CHANNEL),
                    (None, "", TAINT_INCOMPLETE, TAINT_NO_CHANNEL),
                ]
            )
        else:
            drained = _server.close()
            taints = _server.taints
            if not drained:
                safe_print(
                    "[cbts] some subscribers did not leave before the deadline; their "
                    "coverage may be incomplete"
                )
            _record_taints(taints)
            diagnostics = _server.diagnostics
            if diagnostics:
                hooks.active.record_process_diagnostics(diagnostics)
            _server = None
        # Every outcome is recorded by now: pytest_runtest_makereport runs for each
        # test before this hook. Saving here puts the session's whole record --
        # touches, outcomes and taints -- on disk without depending on the atexit
        # save being reached, which a hard exit would skip.
        hooks.active.flush_coverage()
    except Exception as exc:
        safe_print(f"[cbts] session teardown failed: {exc!r}")
        # Stage-scoped, same reasoning as the no-channel branch above: a producer-side
        # session failure has no way to name which tests it affects. Taint first, in
        # case flush_coverage() itself was what failed -- the taint still lands in
        # _tracker's in-memory set either way, for the atexit save to pick up later.
        with contextlib.suppress(Exception):
            hooks.active.record_channel_taints(
                [
                    (None, "", TAINT_ATTRIBUTION, TAINT_SESSION_TEARDOWN_FAILED),
                    (None, "", TAINT_INCOMPLETE, TAINT_SESSION_TEARDOWN_FAILED),
                ]
            )


def _record_taints(taints: Iterable[tuple[Optional[str], str, str, str]]) -> None:
    """Log coverage the channel could not vouch for and store it beside the leaf data.

    Deliberately *not* guarded here: ``record_channel_taints`` is itself the
    mechanism for folding a known problem into the report, so this one call has no
    narrower fallback worth trying on its own failure. Left to propagate to the
    caller (``pytest_sessionfinish``), whose own outer guard both reports it and
    records a broader, session-level taint in its place -- so even losing these
    specific taints still leaves the report self-tainted, just under a more generic
    reason than the one that failed to record.
    """
    if not taints:
        return
    for process_uid, test, kind, reason in taints:
        who = "<this process>" if process_uid is None else (process_uid or "<unidentified>")
        safe_print(
            f"[cbts] tainted coverage ({kind}): {reason} for {test or '<whole stage>'} in {who}"
        )
    # Folded into this process's leaf database, which its final save writes after
    # this hook returns.
    hooks.active.record_channel_taints(taints)
