# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fallback wiring of the session-reuse and session-prefetch plugins.

tests/unittest and tests/integration/defs load the plugins through their own
pytest.ini / conftest; their rootdir sits below this file, so this conftest
is never collected there (no double registration). Any other directory under
tests/ (current or future) picks the hooks up from here, so MPI session
reuse and prefetch cover every test under tests/.

Both plugins define same-named hooks, and ``pytest_plugins`` is only allowed
in a rootdir conftest (this file is not one when pytest runs from the repo
root) — so dispatch to both modules explicitly instead of importing their
hook functions into this namespace (the second import would silently shadow
the first).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # make test_common importable

from test_common import session_prefetcher_hooks as _prefetch  # noqa: E402
from test_common import session_reuse_hooks as _reuse  # noqa: E402


def pytest_configure(config):
    _reuse.pytest_configure(config)
    _prefetch.pytest_configure(config)


def pytest_runtest_setup(item):
    _reuse.pytest_runtest_setup(item)
    _prefetch.pytest_runtest_setup(item)


def pytest_runtest_logreport(report):
    # Reuse's failure fence (drain pools after a failed test); previously not
    # wired in the fallback dirs at all.
    _reuse.pytest_runtest_logreport(report)


def pytest_runtest_logfinish(nodeid, location):
    _reuse.pytest_runtest_logfinish(nodeid, location)


def pytest_sessionfinish(session, exitstatus):
    _reuse.pytest_sessionfinish(session, exitstatus)
    _prefetch.pytest_sessionfinish(session, exitstatus)
