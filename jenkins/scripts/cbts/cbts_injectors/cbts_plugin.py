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

import os
import sys
from typing import Iterable, Optional

import pytest
from cbts.coverage.collection import hooks
from cbts.coverage.collection.channel import (
    ADDRESS_ENV,
    TAINT_ATTRIBUTION,
    TAINT_INCOMPLETE,
    TAINT_NO_CHANNEL,
    ChannelError,
    ContextServer,
)
from cbts.coverage.collection.pool import install_expected_workers_patch

_server: Optional[ContextServer] = None


def pytest_configure(config: pytest.Config) -> None:  # noqa: D401 - pytest hook
    """Install the pool accounting patch and open the context channel."""
    del config
    global _server
    install_expected_workers_patch()
    try:
        _server = ContextServer.start()
    except ChannelError as exc:
        # Report only: CBTS setup failure doesn't block test run
        cause = exc.__cause__
        print(
            f"[cbts] {exc}: {cause!r}; subprocesses keep their inherited CBTS_TEST_ID",
            file=sys.stderr,
        )
        return
    # Set before any test can spawn: ordinary children inherit it at exec, and the patched
    # MPIPoolExecutor reads it here to forward into its workers' env payload.
    os.environ[ADDRESS_ENV] = _server.address


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Optional[pytest.Item]):  # noqa: D401
    """Switch this process to the current test, then bring every subscriber with it."""
    del nextitem
    nodeid = item.nodeid

    # The bootstrap default for a process that starts before it can subscribe.
    os.environ["CBTS_TEST_ID"] = nodeid

    hooks.active.switch_test_context(nodeid)

    if _server is not None and not _server.announce(nodeid):
        print(
            f"[cbts] not all subscribers acknowledged {nodeid}; their next touches may "
            "land on the previous test",
            file=sys.stderr,
        )

    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):  # noqa: D401
    """Record each test's outcome so the merge can flag coverage that isn't safe to trust."""
    del call
    outcome = yield
    report = outcome.get_result()
    # The call phase is the test body; a non-passing setup or teardown is the test's
    # effective outcome (a failing teardown downgrades an already-recorded pass).
    if report.when == "call" or (
        report.when in ("setup", "teardown") and report.outcome != "passed"
    ):
        hooks.active.record_test_outcome(item.nodeid, report.outcome)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: D401
    """Close the channel and wait for every subscriber to save and leave."""
    del session, exitstatus
    global _server
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
            print(
                "[cbts] some subscribers did not leave before the deadline; their "
                "coverage may be incomplete",
                file=sys.stderr,
            )
        _record_taints(taints)
        _server = None
    # Every outcome is recorded by now: pytest_runtest_makereport runs for each test
    # before this hook. Saving here puts the session's whole record -- touches,
    # outcomes and taints -- on disk without depending on the atexit save being
    # reached, which a hard exit would skip.
    hooks.active.flush_coverage()


def _record_taints(taints: Iterable[tuple[Optional[str], str, str, str]]) -> None:
    """Log coverage the channel could not vouch for and store it beside the leaf data."""
    if not taints:
        return
    for process_uid, test, kind, reason in taints:
        who = "<this process>" if process_uid is None else (process_uid or "<unidentified>")
        print(
            f"[cbts] tainted coverage ({kind}): {reason} for {test or '<whole stage>'} in {who}",
            file=sys.stderr,
        )
    # Folded into this process's leaf database, which its final save writes after
    # this hook returns. Not guarded: a failure here loses the record of a problem
    # the channel already reported, and pytest renders the traceback.
    hooks.active.record_channel_taints(taints)
