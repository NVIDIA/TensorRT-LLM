# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VisualGen part of the private-access lint.

Pull requests that change only VisualGen files are claimed by CBTS Tier 1 and run
only test-db entries whose path contains visual_gen/. This file applies the rule
and baseline of tests/unittest/serve/test_private_access_lint.py to
tensorrt_llm/serve/api/visual_gen/, so those pull requests are still checked.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_MAIN = Path(__file__).resolve().parents[2] / "test_private_access_lint.py"
_spec = importlib.util.spec_from_file_location("_serve_private_access_lint", _MAIN)
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)


def test_no_new_cross_package_private_access_in_visual_gen():
    rows = lint.scan(prefixes=(lint.VISUAL_GEN_SCOPE,))
    errors = lint._compare(rows, lint.load_baseline(), prefix=lint.VISUAL_GEN_SCOPE)
    assert not errors, "\n".join(errors)
