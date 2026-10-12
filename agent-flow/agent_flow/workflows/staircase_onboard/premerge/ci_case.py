# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a test-db line the way CI runs it, and record what it measured.

The difference between "the gate passed" and "CI will pass" was, in the run
this was built from, four accuracy cases collected and then **skipped** on
every CI stage: no stage exported the switch the gate had exported. Running
the case as CI does — not as the gate does — is what found it.

What "as CI does" means here (``jenkins/L0_Test.groovy``,
``getPytestBaseCommandLine``):

- the test-db line itself, ``TIMEOUT (n)`` included, is the ``--test-list``;
- CI's pytest flags, from ``tests/integration/defs`` as the working
  directory, with the stage name as ``--test-prefix`` and the repository's
  waives file;
- nothing exported beyond what CI exports (``LLM_ROOT``,
  ``LLM_BACKEND_ROOT``, ``LLM_MODELS_ROOT``, ``MODEL_CACHE_DIR``,
  ``COLUMNS``); the case's own environment travels with the case.

What differs is stated in the generated script: how ranks are launched
(``--launcher``, e.g. ``mpirun -n 4 ... trtllm-llmapi-launch`` where the
container forbids ``MPI_Comm_spawn``) and where the models root points.

The script prints ``HEAD``, the branch and the number of modified tracked
files at start **and** at end, plus where ``tensorrt_llm`` resolved: a result
is only evidence for the tree it measured, and a branch that moved while the
case ran has no single tree.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

TEST_DB_DIR = "tests/integration/test_lists/test-db"
DEFS_DIR = "tests/integration/defs"
WAIVES = "tests/integration/test_lists/waives.txt"
STAGE_MAPPER = "scripts/test_to_stage_mapping.py"

#: CI's own pytest flags, in CI's order, minus the reporting plugins that
#: only make sense inside Jenkins. ``{...}`` fields are filled per case.
CI_PYTEST_FLAGS: tuple[str, ...] = (
    "-vv",
    "-rA",
    "--timeout-method=thread",
    "--apply-test-list-correction",
    "--timeout={timeout}",
    "--rootdir={defs}",
    "--test-prefix={stage}",
    "--waives-file={waives}",
    "--output-dir={out}/",
    "--csv={out}/report.csv",
    "-o",
    "junit_logging=out-err",
    "-p",
    "no:cacheprovider",
    "--unittest-markexpr={markexpr}",
)

#: Variables a cluster session carries and CI's job does not; an MPI launcher
#: that inherits them attaches to the allocation instead of starting its own
#: world, and an abnormal exit then takes the allocation down with it.
SCRUBBED_ENV: tuple[str, ...] = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_TASK_PID",
    "SLURM_STEP_NUM_TASKS",
)


@dataclass(frozen=True)
class DbEntry:
    """One test-db line, with the block it sits in."""

    yml: str
    line: str
    test_id: str
    terms: dict[str, Any] = field(default_factory=dict)
    system_gpu_count: tuple[int | None, int | None] = (None, None)

    @property
    def timeout_minutes(self) -> int | None:
        match = re.search(r"TIMEOUT\s*\((\d+)\)", self.line)
        return int(match.group(1)) if match else None


def _blocks(data: Any) -> list[dict[str, Any]]:
    """The ``{condition, tests}`` blocks of one test-db file, whatever its top level is."""
    if isinstance(data, dict):
        out: list[dict[str, Any]] = []
        for key, value in data.items():
            if key != "version":
                out.extend(_blocks(value))
        return out
    if isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return [{"tests": data}]
        return [item for item in data if isinstance(item, dict)]
    return []


def find_test_db_entries(repo: str | Path, pattern: str) -> list[DbEntry]:
    """Every test-db line whose test id contains ``pattern``."""
    found: list[DbEntry] = []
    for yml in sorted((Path(repo) / TEST_DB_DIR).glob("*.yml")):
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        for block in _blocks(data):
            condition = block.get("condition") or {}
            if not isinstance(condition, dict):
                condition = {}
            gpus = ((condition.get("ranges") or {}).get("system_gpu_count")) or {}
            for line in block.get("tests") or []:
                line = str(line).strip()
                test_id = line.split(" TIMEOUT", 1)[0].strip()
                if pattern in test_id:
                    found.append(
                        DbEntry(
                            yml=yml.name,
                            line=line,
                            test_id=test_id,
                            terms=dict(condition.get("terms") or {}),
                            system_gpu_count=(gpus.get("gte"), gpus.get("lte")),
                        )
                    )
    return found


def resolve_stages(repo: str | Path, test_id: str, *, python: str = sys.executable) -> list[str]:
    """The CI stages that run ``test_id``, from the repository's own mapper.

    Empty when the mapper is missing or cannot answer; the caller then needs
    an explicit ``--stage``.
    """
    mapper = Path(repo) / STAGE_MAPPER
    if not mapper.is_file():
        return []
    try:
        out = subprocess.run(
            [python, str(mapper), "--tests", test_id],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [
        line.strip() for line in out.stdout.splitlines() if re.match(r"^[\w.-]+$", line.strip())
    ]


@dataclass(frozen=True)
class CiCase:
    """Everything needed to run one test-db line as CI would."""

    tree: str
    entry: DbEntry
    stage: str
    out_dir: str
    models_root: str = ""
    launcher: str = ""
    timeout_s: int = 3600
    collect_only: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)
    #: Put the tree itself on ``PYTHONPATH``: needed for a worktree whose
    #: package is not the installed one. Empty unsets ``PYTHONPATH``, as CI.
    pythonpath: str = ""

    @property
    def markexpr(self) -> str:
        return "cpu_only" if self.stage.startswith("CPU-") else "not cpu_only"


def pytest_argv(case: CiCase, test_list: str) -> list[str]:
    """The pytest command line CI would build for this case."""
    tree = case.tree.rstrip("/")
    fields = {
        "timeout": str(case.timeout_s),
        "defs": f"{tree}/{DEFS_DIR}",
        "stage": case.stage,
        "waives": f"{tree}/{WAIVES}",
        "out": case.out_dir.rstrip("/"),
        "markexpr": case.markexpr,
    }
    argv = ["pytest", *(flag.format(**fields) for flag in CI_PYTEST_FLAGS)]
    if case.collect_only:
        argv = ["pytest", "--collect-only", "-q", *argv[1:]]
    return [*argv, f"--test-list={test_list}"]


def render_script(case: CiCase) -> str:
    """A self-contained bash script that runs ``case`` and records its tree."""
    tree = case.tree.rstrip("/")
    out = case.out_dir.rstrip("/")
    test_list = f"{out}/test_list.txt"
    argv = pytest_argv(case, test_list)
    launch = shlex.split(case.launcher) if case.launcher and not case.collect_only else []
    command = " ".join(shlex.quote(a) for a in [*launch, *argv])
    if launch:
        command = "env " + " ".join(f"-u {v}" for v in SCRUBBED_ENV) + " " + command
    exports = {
        "LLM_ROOT": tree,
        "LLM_BACKEND_ROOT": f"{tree}/triton_backend",
        "COLUMNS": "300",
        "PYTHONUNBUFFERED": "1",
    }
    if case.models_root:
        exports["LLM_MODELS_ROOT"] = case.models_root
        exports["MODEL_CACHE_DIR"] = case.models_root
    exports.update(case.extra_env)
    export_lines = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in exports.items())
    pythonpath_line = (
        f"export PYTHONPATH={shlex.quote(case.pythonpath)}"
        if case.pythonpath
        else "unset PYTHONPATH"
    )
    record = shlex.quote(f"{out}/ci_case.json")
    head_probe = (
        f'"HEAD=$(git -C {shlex.quote(tree)} rev-parse --short=10 HEAD) '
        f"branch=$(git -C {shlex.quote(tree)} rev-parse --abbrev-ref HEAD) "
        f'dirty=$(git -C {shlex.quote(tree)} status --porcelain --untracked-files=no | wc -l)"'
    )
    return f"""#!/usr/bin/env bash
# Generated by `staircase-onboard premerge ci`. Runs one test-db line as CI does.
#   test-db: {case.entry.yml}
#   line:    {case.entry.line}
#   stage:   {case.stage}
# Differs from CI in: launcher={case.launcher or "(none: pytest directly)"};
# LLM_MODELS_ROOT={case.models_root or "(inherited)"}.
set -u
mkdir -p {shlex.quote(out)}
printf '%s\\n' {shlex.quote(case.entry.line)} > {shlex.quote(test_list)}
{export_lines}
{pythonpath_line}
cd {shlex.quote(tree + "/" + DEFS_DIR)} || exit 2
START_TREE={head_probe}
echo "CI_CASE_START $(date -u +%FT%TZ) case={case.entry.test_id} stage={case.stage} $START_TREE host=$(hostname)"
python -c "import importlib.util as u; print('CI_CASE tensorrt_llm resolves to', u.find_spec('tensorrt_llm').origin)"
{command}
rc=$?
END_TREE={head_probe}
echo "CI_CASE_END $(date -u +%FT%TZ) rc=$rc $END_TREE"
python3 - "$rc" "$START_TREE" "$END_TREE" <<'PY' > {record}
import json, sys
rc, start, end = sys.argv[1:4]
print(json.dumps({{"rc": int(rc), "start": start, "end": end,
                  "same_tree": start == end, "line": {json.dumps(case.entry.line)},
                  "stage": {json.dumps(case.stage)}}}, indent=2))
PY
exit $rc
"""


def write_case(case: CiCase) -> Path:
    """Write the script (and the case description) into ``case.out_dir``."""
    out = Path(case.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    script = out / "run_ci_case.sh"
    script.write_text(render_script(case), encoding="utf-8")
    script.chmod(0o755)
    (out / "case.json").write_text(json.dumps(asdict(case), indent=2) + "\n", encoding="utf-8")
    return script


def case_dir_name(entry: DbEntry) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "-", entry.test_id.rsplit("::", 1)[-1]).strip("-")
    return f"ci-{tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def build_cases(
    tree: str | Path,
    pattern: str,
    *,
    out_root: str | Path,
    stage: str = "",
    models_root: str = "",
    launcher: str = "",
    collect_only: bool = False,
    pythonpath: str = "",
    python: str = sys.executable,
) -> list[CiCase]:
    """One :class:`CiCase` per matching test-db line."""
    cases = []
    for entry in find_test_db_entries(tree, pattern):
        chosen = stage or next(iter(resolve_stages(tree, entry.test_id, python=python)), "")
        if not chosen:
            chosen = "UNKNOWN-STAGE"
        cases.append(
            CiCase(
                tree=str(tree),
                entry=entry,
                stage=chosen,
                out_dir=str(Path(out_root) / case_dir_name(entry)),
                models_root=models_root,
                launcher=launcher,
                collect_only=collect_only,
                pythonpath=pythonpath,
            )
        )
    return cases


def describe(entries: Sequence[DbEntry]) -> str:
    return "\n".join(f"{e.yml}: {e.line}  terms={e.terms}" for e in entries)


__all__ = [
    "CI_PYTEST_FLAGS",
    "CiCase",
    "DbEntry",
    "build_cases",
    "describe",
    "find_test_db_entries",
    "pytest_argv",
    "render_script",
    "resolve_stages",
    "write_case",
]
