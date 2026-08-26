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
``cbts_pool``.
"""

import os
import sys

import pytest

from cbts_channel import ADDRESS_ENV, ContextServer
from cbts_pool import install_expected_workers_patch, sitecustomize_call

_server: ContextServer | None = None


def pytest_configure(config):  # noqa: D401 - pytest hook
    """Install the pool accounting patch and open the context channel."""
    del config
    global _server
    install_expected_workers_patch()
    _server = ContextServer.start()
    if _server is None:
        print("[cbts] context channel unavailable; subprocesses keep their inherited "
              "CBTS_TEST_ID", file=sys.stderr)
        return
    # Set before any test can spawn: ordinary children inherit it at exec, and the patched
    # MPIPoolExecutor reads it here to forward into its workers' env payload.
    os.environ[ADDRESS_ENV] = _server.address


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # noqa: D401 - pytest hook
    """Switch this process to the current test, then bring every subscriber with it."""
    del nextitem
    nodeid = item.nodeid

    # The bootstrap default for a process that starts before it can subscribe.
    os.environ["CBTS_TEST_ID"] = nodeid

    sitecustomize_call("switch_test_context", nodeid)

    if _server is not None and not _server.announce(nodeid):
        print(f"[cbts] not all subscribers acknowledged {nodeid}; their next touches may "
              "land on the previous test", file=sys.stderr)

    yield


def pytest_sessionfinish(session, exitstatus):  # noqa: D401 - pytest hook
    """Close the channel and wait for every subscriber to save and leave."""
    del session, exitstatus
    global _server
    if _server is None:
        return
    if not _server.close():
        print("[cbts] some subscribers did not leave before the deadline; their coverage "
              "may be incomplete", file=sys.stderr)
    _server = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: D401 - pytest hook
    """Record each test's outcome so the merge can flag coverage that isn't safe to trust."""
    del call
    outcome = yield
    report = outcome.get_result()
    # The call phase is the test body; a non-passing setup or teardown is the test's
    # effective outcome (a failing teardown downgrades an already-recorded pass).
    if report.when == "call" or (
        report.when in ("setup", "teardown") and report.outcome != "passed"
    ):
        sitecustomize_call("record_test_outcome", item.nodeid, report.outcome)
