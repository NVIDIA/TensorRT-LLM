#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for scripts/legacy_utils.py.

Covers:
- Behavioral tests for compare_against_baseline() (8 tests)
- Migrated structural/behavioral tests from test_ruff_legacy_lint_baseline.py
- Migrated tests from test_generate_legacy_lint_config.py
- CLI subcommand tests
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "legacy_utils.py"
TEMPLATES_DIR = REPO_ROOT / "scripts" / "templates"


@pytest.fixture()
def mod():
    """Import the unified script as a module."""
    spec = importlib.util.spec_from_file_location("legacy_utils", SCRIPT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ==========================================================================
# Behavioral tests: compare_against_baseline()
# ==========================================================================


class TestCompareAgainstBaseline:
    def test_regression_detected(self, mod):
        """Count > baseline -> regression with correct delta."""
        baseline = {"file.py": {"E501": 2}}
        current = {"file.py": {"E501": 5}}
        checked = {"file.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == [("file.py", "E501", 3)]
        assert improvements == []

    def test_no_regression_at_baseline(self, mod):
        """Count == baseline -> empty regressions."""
        baseline = {"file.py": {"E501": 3}}
        current = {"file.py": {"E501": 3}}
        checked = {"file.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == []
        assert improvements == []

    def test_improvement_detected(self, mod):
        """Count < baseline -> improvement with correct delta."""
        baseline = {"file.py": {"E501": 5}}
        current = {"file.py": {"E501": 2}}
        checked = {"file.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == []
        assert improvements == [("file.py", "E501", 3)]

    def test_new_file_not_in_baseline(self, mod):
        """File with violations, empty baseline -> regression."""
        baseline = {}
        current = {"new.py": {"E501": 1}}
        checked = {"new.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == [("new.py", "E501", 1)]
        assert improvements == []

    def test_file_not_checked(self, mod):
        """File in baseline but not in checked_files -> no output."""
        baseline = {"other.py": {"E501": 10}}
        current = {}
        checked = {"file.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == []
        assert improvements == []

    def test_all_violations_fixed(self, mod):
        """Baseline file, zero current -> improvements for each rule."""
        baseline = {"file.py": {"E501": 3, "E711": 2}}
        current = {}
        checked = {"file.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == []
        assert sorted(improvements) == [("file.py", "E501", 3), ("file.py", "E711", 2)]

    def test_new_rule_not_in_baseline(self, mod):
        """Existing file, new rule code -> regression."""
        baseline = {"file.py": {"E501": 1}}
        current = {"file.py": {"E501": 1, "E711": 3}}
        checked = {"file.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == [("file.py", "E711", 3)]
        assert improvements == []

    def test_meta_key_ignored(self, mod):
        """_meta dict in baseline doesn't interfere."""
        baseline = {
            "_meta": {"total_violations": 10},
            "file.py": {"E501": 2},
        }
        current = {"file.py": {"E501": 2}}
        checked = {"file.py"}
        regressions, improvements = mod.compare_against_baseline(current, baseline, checked)
        assert regressions == []
        assert improvements == []


# ==========================================================================
# Migrated from test_ruff_legacy_lint_baseline.py
# ==========================================================================


class TestDocstring:
    def test_docstring_mentions_auto_fix_failure(self, mod):
        """Module docstring should mention auto-modified / re-staging."""
        docstring = mod.__doc__
        assert docstring is not None
        assert "auto-modified" in docstring or "re-staging" in docstring, (
            f"Docstring must mention auto-fix failure mode, got:\n{docstring}"
        )


class TestRunRuffJsonMerged:
    def test_run_ruff_json_no_fix_removed(self, mod):
        """run_ruff_json_no_fix should no longer exist."""
        assert not hasattr(mod, "run_ruff_json_no_fix")

    def test_run_ruff_json_fix_flag_true(self, mod):
        """run_ruff_json(files, fix=True) should include --fix in the command."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="[]", stderr="")
            mod.run_ruff_json(["dummy.py"], fix=True)
            cmd = mock_run.call_args[0][0]
            assert "--fix" in cmd

    def test_run_ruff_json_fix_flag_false(self, mod):
        """run_ruff_json(files, fix=False) should NOT include --fix."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="[]", stderr="")
            mod.run_ruff_json(["dummy.py"], fix=False)
            cmd = mock_run.call_args[0][0]
            assert "--fix" not in cmd

    def test_run_ruff_json_default_is_fix(self, mod):
        """run_ruff_json default should be fix=True."""
        sig = inspect.signature(mod.run_ruff_json)
        assert sig.parameters["fix"].default is True


class TestNoGetRepoRoot:
    def test_no_get_repo_root_function(self, mod):
        """get_repo_root should not exist; use REPO_ROOT constant."""
        assert not hasattr(mod, "get_repo_root")

    def test_precommit_mode_uses_repo_root_constant(self, mod):
        """precommit_mode should use REPO_ROOT, not call get_repo_root()."""
        source = inspect.getsource(mod.precommit_mode)
        assert "get_repo_root()" not in source

    def test_update_baseline_mode_uses_repo_root_constant(self, mod):
        """update_baseline_mode should use REPO_ROOT, not call get_repo_root()."""
        source = inspect.getsource(mod.update_baseline_mode)
        assert "get_repo_root()" not in source


class TestNormalizePathUsage:
    def test_precommit_mode_uses_normalize_path(self, mod):
        """precommit_mode should use normalize_path, not inline p.relative_to(repo_root)."""
        source = inspect.getsource(mod.precommit_mode)
        assert "p.relative_to(repo_root)" not in source


# ==========================================================================
# Migrated from test_generate_legacy_lint_config.py
# ==========================================================================


class TestArgparseDefault:
    def test_check_flag_does_not_set_generate(self, mod):
        """When check-configs is the subcommand, gen-configs shouldn't also be active."""
        parser = mod._build_parser()
        args = parser.parse_args(["check-configs"])
        assert args.command == "check-configs"

    def test_no_flags_shows_help(self, mod):
        """When no subcommand is given, command should be None."""
        parser = mod._build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestReplaceBlockNoTrailingNewline:
    def test_no_trailing_newline_does_not_raise(self, mod):
        """replace_managed_block should not raise ValueError when no trailing newline after END marker."""
        content = f"before\n{mod.BEGIN_MARKER}\nold content\n{mod.END_MARKER}"
        new_block = f"{mod.BEGIN_MARKER}\nnew content\n{mod.END_MARKER}\n"
        result = mod.replace_managed_block(content, new_block)
        assert "new content" in result

    def test_with_trailing_newline_still_works(self, mod):
        """Normal case: content has newline after END marker."""
        content = f"before\n{mod.BEGIN_MARKER}\nold content\n{mod.END_MARKER}\nafter\n"
        new_block = f"{mod.BEGIN_MARKER}\nnew content\n{mod.END_MARKER}\n"
        result = mod.replace_managed_block(content, new_block)
        assert "new content" in result
        assert result.endswith("after\n")


class TestAtomicWrites:
    def test_do_generate_no_partial_writes_on_marker_failure(self, mod, tmp_path):
        """If a marker is missing in one file, no files should be modified."""
        paths = ["a.py", "b.py"]

        good_content = f"start\n{mod.BEGIN_MARKER}\nold\n{mod.END_MARKER}\nend\n"
        bad_content = "no markers here"

        toml_file = tmp_path / "ruff-legacy.toml"
        pyproject_file = tmp_path / "pyproject.toml"
        precommit_file = tmp_path / ".pre-commit-config.yaml"

        pyproject_file.write_text(good_content)
        precommit_file.write_text(bad_content)

        orig_toml = mod.RUFF_LEGACY_TOML
        orig_pyproject = mod.PYPROJECT_TOML
        orig_precommit = mod.PRECOMMIT_YAML
        try:
            mod.RUFF_LEGACY_TOML = toml_file
            mod.PYPROJECT_TOML = pyproject_file
            mod.PRECOMMIT_YAML = precommit_file

            with (
                mock.patch.object(mod, "generate_ruff_legacy_toml", return_value="toml"),
                mock.patch.object(mod, "generate_pyproject_block", return_value="pyblock"),
                mock.patch.object(mod, "generate_precommit_block", return_value="preblock"),
            ):
                with pytest.raises(ValueError):
                    mod.do_generate(paths)

            assert not toml_file.exists()
            assert pyproject_file.read_text() == good_content
        finally:
            mod.RUFF_LEGACY_TOML = orig_toml
            mod.PYPROJECT_TOML = orig_pyproject
            mod.PRECOMMIT_YAML = orig_precommit


# ==========================================================================
# Migrated from test_ruff_legacy_toml_template.py
# ==========================================================================


@pytest.fixture()
def rendered_toml():
    """Render ruff-legacy.toml.j2 with a minimal paths list."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template("ruff-legacy.toml.j2")
    return template.render(paths=["dummy/file.py"])


class TestNoAutoDeployPerFileIgnore:
    def test_no_auto_deploy_in_template(self, rendered_toml):
        """Rendered ruff-legacy.toml should not contain auto_deploy per-file-ignore."""
        assert "auto_deploy" not in rendered_toml


class TestNoPylintMaxArgs:
    def test_no_max_args_in_template(self, rendered_toml):
        """Rendered ruff-legacy.toml should not contain max-args config."""
        assert "max-args" not in rendered_toml

    def test_no_pylint_section_in_template(self, rendered_toml):
        """Rendered ruff-legacy.toml should not contain [lint.pylint] section."""
        assert "[lint.pylint]" not in rendered_toml


# ==========================================================================
# CLI subcommand tests
# ==========================================================================


class TestArgparseCLI:
    def test_argparse_help(self):
        """--help should exit 0 and print usage."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()

    def test_argparse_unknown_flag_errors(self):
        """Unknown flags like --bogus should cause exit code 2."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--bogus"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_argparse_lint_precommit(self, mod):
        """lint-precommit subcommand should be recognized with files."""
        parser = mod._build_parser()
        args = parser.parse_args(["lint-precommit", "a.py", "b.py"])
        assert args.command == "lint-precommit"
        assert args.files == ["a.py", "b.py"]

    def test_argparse_lint_update_violations(self, mod):
        """lint-update-violations subcommand should be recognized."""
        parser = mod._build_parser()
        args = parser.parse_args(["lint-update-violations"])
        assert args.command == "lint-update-violations"

    def test_argparse_gen_configs(self, mod):
        """gen-configs subcommand should be recognized."""
        parser = mod._build_parser()
        args = parser.parse_args(["gen-configs"])
        assert args.command == "gen-configs"

    def test_argparse_check_configs(self, mod):
        """check-configs subcommand should be recognized."""
        parser = mod._build_parser()
        args = parser.parse_args(["check-configs"])
        assert args.command == "check-configs"

    def test_argparse_prune_files(self, mod):
        """prune-files subcommand should be recognized."""
        parser = mod._build_parser()
        args = parser.parse_args(["prune-files"])
        assert args.command == "prune-files"

    def test_argparse_no_command_returns_none(self, mod):
        """No subcommand -> command is None."""
        parser = mod._build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ==========================================================================
# Error handling: exceptions instead of sys.exit()
# ==========================================================================


class TestExceptionBasedErrors:
    def test_run_ruff_json_raises_ruff_error(self, mod):
        """run_ruff_json raises RuffError on unexpected ruff failure."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="ruff broke")
            with pytest.raises(mod.RuffError):
                mod.run_ruff_json(["dummy.py"])

    def test_run_ruff_json_raises_on_bad_json(self, mod):
        """run_ruff_json raises RuffError on invalid JSON output."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="not json", stderr="")
            with pytest.raises(mod.RuffError):
                mod.run_ruff_json(["dummy.py"])

    def test_load_baseline_raises_file_not_found(self, mod, tmp_path):
        """load_baseline raises FileNotFoundError when file is missing."""
        orig = mod.BASELINE_PATH
        try:
            mod.BASELINE_PATH = tmp_path / "nonexistent.json"
            with pytest.raises(FileNotFoundError):
                mod.load_baseline()
        finally:
            mod.BASELINE_PATH = orig

    def test_load_baseline_strips_meta(self, mod, tmp_path):
        """load_baseline should strip _meta key from returned dict."""
        import json

        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps({"_meta": {"total": 10}, "file.py": {"E501": 2}}))
        orig = mod.BASELINE_PATH
        try:
            mod.BASELINE_PATH = baseline_file
            result = mod.load_baseline()
            assert "_meta" not in result
            assert "file.py" in result
        finally:
            mod.BASELINE_PATH = orig

    def test_read_legacy_files_raises_file_not_found(self, mod, tmp_path):
        """read_legacy_files raises FileNotFoundError when file is missing."""
        orig = mod.LEGACY_FILES_TXT
        try:
            mod.LEGACY_FILES_TXT = tmp_path / "nonexistent.txt"
            with pytest.raises(FileNotFoundError):
                mod.read_legacy_files()
        finally:
            mod.LEGACY_FILES_TXT = orig


# ==========================================================================
# Tests for identify_new_violations()
# ==========================================================================


class TestIdentifyNewViolations:
    def test_single_regression_line_and_message(self, mod):
        """Single violation above baseline returns (filepath, line, code, message)."""
        baseline = {"src/a.py": {"E722": 0}}
        violations = [
            {
                "filename": "/repo/src/a.py",
                "code": "E722",
                "message": "Do not use bare `except`",
                "location": {"row": 42, "column": 1},
            },
        ]
        result = mod.identify_new_violations(violations, baseline, Path("/repo"))
        assert result == [("src/a.py", 42, "E722", "Do not use bare `except`")]

    def test_multiple_regressions_sorted(self, mod):
        """Multiple violations across files are sorted by filepath then line."""
        baseline = {
            "src/a.py": {"E722": 0},
            "src/b.py": {"E501": 0},
        }
        violations = [
            {
                "filename": "/repo/src/b.py",
                "code": "E501",
                "message": "Line too long",
                "location": {"row": 10, "column": 1},
            },
            {
                "filename": "/repo/src/a.py",
                "code": "E722",
                "message": "Do not use bare `except`",
                "location": {"row": 50, "column": 1},
            },
            {
                "filename": "/repo/src/a.py",
                "code": "E722",
                "message": "Do not use bare `except`",
                "location": {"row": 20, "column": 1},
            },
        ]
        result = mod.identify_new_violations(violations, baseline, Path("/repo"))
        assert result == [
            ("src/a.py", 20, "E722", "Do not use bare `except`"),
            ("src/a.py", 50, "E722", "Do not use bare `except`"),
            ("src/b.py", 10, "E501", "Line too long"),
        ]

    def test_no_regressions_returns_empty(self, mod):
        """Violations at or below baseline return empty list."""
        baseline = {"src/a.py": {"E722": 2}}
        violations = [
            {
                "filename": "/repo/src/a.py",
                "code": "E722",
                "message": "Do not use bare `except`",
                "location": {"row": 10, "column": 1},
            },
        ]
        result = mod.identify_new_violations(violations, baseline, Path("/repo"))
        assert result == []

    def test_new_file_not_in_baseline_shows_all(self, mod):
        """Violations in a file not in baseline are all returned."""
        baseline = {}
        violations = [
            {
                "filename": "/repo/new.py",
                "code": "E501",
                "message": "Line too long",
                "location": {"row": 5, "column": 1},
            },
            {
                "filename": "/repo/new.py",
                "code": "E722",
                "message": "Do not use bare `except`",
                "location": {"row": 12, "column": 1},
            },
        ]
        result = mod.identify_new_violations(violations, baseline, Path("/repo"))
        assert result == [
            ("new.py", 5, "E501", "Line too long"),
            ("new.py", 12, "E722", "Do not use bare `except`"),
        ]

    def test_syntax_errors_excluded(self, mod):
        """Violations with code=None (syntax errors) are excluded."""
        baseline = {}
        violations = [
            {
                "filename": "/repo/src/a.py",
                "code": None,
                "message": "SyntaxError: invalid syntax",
                "location": {"row": 1, "column": 1},
            },
        ]
        result = mod.identify_new_violations(violations, baseline, Path("/repo"))
        assert result == []

    def test_at_baseline_not_shown(self, mod):
        """Exactly at baseline count -> no regressions reported."""
        baseline = {"src/a.py": {"E722": 1}}
        violations = [
            {
                "filename": "/repo/src/a.py",
                "code": "E722",
                "message": "Do not use bare `except`",
                "location": {"row": 7, "column": 1},
            },
        ]
        result = mod.identify_new_violations(violations, baseline, Path("/repo"))
        assert result == []


# ==========================================================================
# Tests for format_improvements()
# ==========================================================================


class TestFormatImprovements:
    def test_single_improvement(self, mod):
        """Single improvement tuple is formatted correctly."""
        improvements = [("src/a.py", "E722", 1)]
        result = mod.format_improvements(improvements)
        assert result == ["  src/a.py  E722 (-1)"]

    def test_multiple_sorted(self, mod):
        """Multiple improvements are sorted by filepath then rule."""
        improvements = [("src/b.py", "E711", 2), ("src/a.py", "E501", 3)]
        result = mod.format_improvements(improvements)
        assert result == [
            "  src/a.py  E501 (-3)",
            "  src/b.py  E711 (-2)",
        ]

    def test_empty_returns_empty(self, mod):
        """Empty input returns empty list."""
        result = mod.format_improvements([])
        assert result == []
