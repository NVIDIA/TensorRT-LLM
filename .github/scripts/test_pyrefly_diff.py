# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for matching advisory Pyrefly diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyrefly_diff import _new_errors, _read_report, _renames, main


def _error(path: str = "module.py", line: int = 1, name: str = "unknown_name") -> dict:
    return {"path": path, "line": line, "name": "unknown-name", "description": f"Unknown {name}"}


class DiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name) / "base"
        self.merged = Path(self.directory.name) / "merged"
        self.base.mkdir()
        self.merged.mkdir()

    def _run_cli(self, error: str = "") -> dict:
        output = Path(self.directory.name) / "output"
        event = Path(self.directory.name) / "event.json"
        event.write_text(
            json.dumps(
                {
                    "pull_request": {
                        "number": 42,
                        "base": {"ref": "release/test"},
                        "head": {"sha": "a" * 40},
                    }
                }
            )
        )
        arguments = [
            "pyrefly_diff.py",
            "--event",
            str(event),
            "--output",
            str(output),
            "--base-root",
            str(self.base),
            "--merged-root",
            str(self.merged),
            "--base-sha",
            "b" * 40,
            "--error",
            error,
        ]
        with patch("sys.argv", arguments):
            main()
        return json.loads((output / "report.json").read_text())

    def test_cli_compares_existing_reports_without_running_the_checker(self) -> None:
        (self.base / "module.py").write_text("unknown_name\n")
        (self.merged / "module.py").write_text("unknown_name\nnew_name\n")
        output = Path(self.directory.name) / "output"
        output.mkdir()
        added = _error(line=2, name="new_name")
        for name, errors in (("base", [_error()]), ("merged", [_error(), added])):
            (output / f"{name}.json").write_text(
                json.dumps(
                    {
                        "pyrefly_exit_code": 1,
                        "result": {"errors": errors},
                    }
                )
            )
        (output / "changes").write_bytes(b"M\0module.py\0")
        report = self._run_cli()
        self.assertEqual(report["base_ref"], "release/test")
        self.assertEqual(report["base_sha"], "b" * 40)
        self.assertEqual(report["errors"], [added])
        self.assertIsNone(report["error"])

    def test_failed_workflow_step_produces_incomplete_report_without_scan_results(self) -> None:
        report = self._run_cli(error="Could not install Pyrefly.")
        self.assertEqual(report["error"], "Could not install Pyrefly.")
        self.assertEqual(report["errors"], [])

    def test_rename_input_preserves_spaces_and_accepts_an_empty_diff(self) -> None:
        changes = Path(self.directory.name) / "changes"
        changes.write_bytes(b"M\0modified.py\0R100\0old name.py\0new name.py\0A\0added.py\0")
        self.assertEqual(_renames(changes), {"old name.py": "new name.py"})
        changes.write_bytes(b"")
        self.assertEqual(_renames(changes), {})

    def test_line_shifts_do_not_report_existing_errors(self) -> None:
        (self.base / "module.py").write_text("unknown_name\n")
        (self.merged / "module.py").write_text("# inserted\n\nunknown_name\nnew_name\n")
        added = _error(line=4, name="new_name")
        result = _new_errors([_error()], [_error(line=3), added], self.base, self.merged, {})
        self.assertEqual(result, [added])

    def test_additional_identical_occurrence_is_new(self) -> None:
        (self.base / "module.py").write_text("unknown_name\n")
        (self.merged / "module.py").write_text("unknown_name\nunknown_name\n")
        result = _new_errors([_error()], [_error(), _error(line=2)], self.base, self.merged, {})
        self.assertEqual(result, [_error(line=2)])

    def test_fixing_one_error_does_not_hide_a_different_new_error(self) -> None:
        (self.base / "module.py").write_text("unknown_name\n")
        (self.merged / "module.py").write_text("different_name\n")
        added = _error(name="different_name")
        self.assertEqual(_new_errors([_error()], [added], self.base, self.merged, {}), [added])

    def test_rename_preserves_existing_errors(self) -> None:
        (self.base / "module.py").write_text("unknown_name\n")
        (self.merged / "renamed.py").write_text("unknown_name\n")
        self.assertEqual(
            _new_errors(
                [_error()],
                [_error(path="renamed.py")],
                self.base,
                self.merged,
                {"module.py": "renamed.py"},
            ),
            [],
        )

    def test_new_diagnostic_in_unchanged_file_is_reported(self) -> None:
        for root in (self.base, self.merged):
            (root / "client.py").write_text("call_api()\n")
        added = {**_error(path="client.py"), "name": "missing-argument"}
        self.assertEqual(_new_errors([], [added], self.base, self.merged, {}), [added])

    def test_advisory_exit_success_does_not_hide_tool_failure(self) -> None:
        report_path = self.base / "report.json"
        for report in (
            {"report_error": "invalid JSON", "pyrefly_exit_code": 0},
            {"pyrefly_exit_code": 137, "result": {"errors": []}},
            {"pyrefly_exit_code": 0, "result": {}},
        ):
            with self.subTest(report=report):
                report_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    _read_report(report_path)

    def test_valid_report_with_diagnostics_is_not_a_tool_failure(self) -> None:
        report_path = self.base / "report.json"
        report_path.write_text(
            json.dumps({"pyrefly_exit_code": 1, "result": {"errors": [_error()]}})
        )
        self.assertEqual(_read_report(report_path), [_error()])


if __name__ == "__main__":
    unittest.main()
