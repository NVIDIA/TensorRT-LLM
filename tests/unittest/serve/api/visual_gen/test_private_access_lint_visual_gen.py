# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VisualGen part of the private-access lint.

Pull requests that change only VisualGen files are claimed by CBTS Tier 1 and run
only test-db entries whose path contains visual_gen/. This file applies the rule
and baseline of tests/unittest/serve/test_private_access_lint.py to the existing
frontends, including the VisualGen handlers in serve/openai_server.py. Keep the
full frontend scope until those handlers move into their own directory.
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
    for scope in lint.SCOPES:
        assert (lint.REPO_ROOT / scope).is_dir(), f"Frontend scope does not exist: {scope}"
    rows = lint.scan(root=lint.REPO_ROOT)
    errors = lint._compare(rows, lint.load_baseline())
    assert not errors, "\n".join(errors)


def test_visual_gen_check_detects_new_frontend_access(tmp_path, monkeypatch):
    for scope in lint.SCOPES:
        (tmp_path / scope).mkdir(parents=True)
    (tmp_path / "tensorrt_llm/serve/openai_server.py").write_text("engine._private_state\n")
    owner = tmp_path / "tensorrt_llm/llmapi/llm.py"
    owner.parent.mkdir()
    owner.write_text("class LLM:\n    _private_state = None\n")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("entries: []\n")
    monkeypatch.setattr(lint, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lint, "BASELINE_FILE", baseline)
    monkeypatch.setattr(lint, "OVERRIDES", {})
    with pytest.raises(AssertionError, match="New private access to `_private_state`"):
        test_no_new_cross_package_private_access_in_visual_gen()
