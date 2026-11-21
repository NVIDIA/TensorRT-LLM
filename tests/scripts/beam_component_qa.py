# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Run explicitly selected component tests and save their pytest reports for manual QA.

Example (run from the repository root with a compatible native build)::

    python tests/scripts/beam_component_qa.py --output /tmp/beam-router-qa -- \
        tests/unittest/_torch/modeling_v2/models/beam/_components/moe_router -v

The output directory must be new. Reports may contain local paths and scheduler IDs;
keep them outside the public repository. This script does not supply source captures,
change test selection or tolerances, or turn skipped qualification into a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


def environment(root: Path) -> dict:
    """Identify the checkout and imported runtime separately, without loading a model."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
        source_revision = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        source_revision = None
    result = {
        "source_revision": source_revision,
        "python": sys.version,
        "machine": platform.machine(),
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    # Do not initialize CUDA just to report a collection failure or a skipped test.
    for name in ("torch", "tensorrt_llm"):
        module = sys.modules.get(name)
        if module is not None:
            result[name] = {"version": module.__version__, "file": module.__file__}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        result["cuda"] = {
            "version": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        }
    return result


class QaReport:
    """Collect standard pytest outcomes and record_property metrics, including failures."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.reports: list[dict] = []

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed or report.skipped:
            self.reports.append(
                {
                    "nodeid": report.nodeid,
                    "phase": "collection",
                    "outcome": report.outcome,
                    "details": str(report.longrepr),
                }
            )

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.reports.append(
            {
                "nodeid": report.nodeid,
                "phase": report.when,
                "outcome": report.outcome,
                "duration": report.duration,
                "properties": report.user_properties,
                "details": str(report.longrepr) if report.longrepr else None,
            }
        )

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        record = {
            "exit_code": int(exitstatus),
            "collect_only": session.config.option.collectonly,
            "tests_collected": session.testscollected,
            "environment": environment(session.config.rootpath),
            "reports": self.reports,
        }
        (self.output / "report.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New local QA report directory")
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="-- followed by pytest selection/options"
    )
    args = parser.parse_args()
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if not pytest_args:
        parser.error("provide an explicit test selection after --")
    args.output.mkdir(parents=True, exist_ok=False)
    return int(pytest.main(pytest_args, plugins=[QaReport(args.output)]))


if __name__ == "__main__":
    raise SystemExit(main())
