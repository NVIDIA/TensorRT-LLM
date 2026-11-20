# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenEngine part of the private-access lint.

Pull requests that change only tensorrt_llm/grpc/openengine/ are claimed by CBTS
Tier 1 and run only the test-db entries under unittest/grpc/openengine/. This file
applies the rule and baseline of tests/unittest/serve/test_private_access_lint.py
to tensorrt_llm/grpc/openengine/, so those pull requests are still checked.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_MAIN = Path(__file__).resolve().parents[2] / "serve" / "test_private_access_lint.py"
_spec = importlib.util.spec_from_file_location("_serve_private_access_lint", _MAIN)
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)


def test_no_new_cross_package_private_access_in_openengine():
    assert (lint.REPO_ROOT / lint.OPENENGINE_SCOPE).is_dir(), (
        f"{lint.OPENENGINE_SCOPE} does not exist. If OpenEngine moved, point OPENENGINE_SCOPE "
        "in tests/unittest/serve/test_private_access_lint.py at the new directory, and move "
        "this file and its l0_cpu.yml entry to the test path the CBTS OpenEngine rule keeps "
        "(jenkins/scripts/cbts/cbts/rules/openengine.py)."
    )
    rows = lint.scan(prefixes=(lint.OPENENGINE_SCOPE,))
    errors = lint._compare(rows, lint.load_baseline(), prefix=lint.OPENENGINE_SCOPE)
    assert not errors, "\n".join(errors)
