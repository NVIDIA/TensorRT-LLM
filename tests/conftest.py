# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fallback wiring of the session-reuse plugin for test dirs WITHOUT an ini.

tests/unittest and tests/integration/defs load the plugin through the ``-p``
option in their own pytest.ini; their rootdir sits below this file, so this
conftest is never collected there (no double registration). Any other
directory under tests/ (current or future) resolves its rootdir at the repo
root and picks the hooks up from here, so automatic MPI session reuse covers
every test under tests/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # make test_common importable

from test_common.session_reuse_hooks import (  # noqa: E402,F401
    pytest_configure,
    pytest_runtest_setup,
    pytest_sessionfinish,
)
