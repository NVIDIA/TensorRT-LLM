#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a tree's pytest tests and record how each ended, with the tree's commit and environment.

Usage, in the tree under test with that tree's python (standard library and
pytest only, so it runs by path in any job)::

    <python> test_driver.py run --cache DIR [--repo TREE] [--selection FILE] [ITEM ...] [-- PYTEST ARGS]

``run`` takes test files and node ids (``ITEM``, or one per line in
``--selection``) and runs one pytest process per test file, a ``session``
each: a file that brings its process down loses only its own results, and a
job cut short keeps every file it finished.

A session writes one JSON record to ``<cache>/sessions/``: the tree's commit
and whether its tracked files were clean, the environment and its digest, the
tests each file collected, and per test its outcome, duration and failure
message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

#: Bumped when a record's meaning changes; the reader skips other versions.
RECORD_VERSION = 1
SESSIONS_DIRNAME = "sessions"
#: Outcomes, the worse the higher: a test's is its worst phase's.
SEVERITY = {None: 0, "passed": 1, "skipped": 2, "xfailed": 2, "xpassed": 3, "failed": 4, "error": 5}

#: Environment variables that change what a test computes, by prefix.
ENV_PREFIXES = (
    "TLLM_",
    "TRTLLM_",
    "TRT_LLM_",
    "TORCH_",
    "PYTORCH_",
    "NCCL_",
    "CUDA_",
    "TRITON_",
    "FLASHINFER_",
    "HF_",
    "TRANSFORMERS_",
)
ENV_NAMES = ("LLM_MODELS_ROOT", "PYTHONHASHSEED")
#: ... less the ones that only say where something is kept or how it is reached.
ENV_SKIP_SUFFIXES = ("_DIR", "_PATH", "_HOME", "_FILE", "_LOG", "_PORT", "_ADDR", "_IFNAME")
ENV_SKIP_NAMES = ("CUDA_VISIBLE_DEVICES",)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env_var_kept(name: str) -> bool:
    if name in ENV_SKIP_NAMES or name.endswith(ENV_SKIP_SUFFIXES):
        return name in ENV_NAMES
    return name in ENV_NAMES or name.startswith(ENV_PREFIXES)


def _gpus() -> tuple[list[str], str]:
    """The GPU models the driver lists, and its version; empty on a host without one."""
    models = []
    root = "/proc/driver/nvidia/gpus"
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        entries = []
    for entry in entries:
        try:
            with open(os.path.join(root, entry, "information"), encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("Model:"):
                        models.append(line.split(":", 1)[1].strip())
        except OSError:
            continue
    driver = ""
    try:
        with open("/proc/driver/nvidia/version", encoding="utf-8") as handle:
            found = re.search(r"Kernel Module\s+(?:for \w+\s+)?([\d.]+)", handle.readline())
            driver = found.group(1) if found else ""
    except OSError:
        pass
    return sorted(models), driver


def environment() -> dict:
    """What, besides the tree, decides a test's result: interpreter, packages, GPUs, settings."""
    import importlib.metadata as metadata
    import platform

    packages = set()
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").strip().lower()
        if name:
            packages.add(f"{name}=={dist.version}")
    models, driver = _gpus()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = (
        len([d for d in visible.split(",") if d.strip()]) if visible is not None else len(models)
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "packages": sorted(packages),
        "gpus": models,
        "visible_gpus": count,
        "driver": driver,
        "vars": {k: v for k, v in sorted(os.environ.items()) if _env_var_kept(k)},
    }


def env_digest(env: dict) -> str:
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _git(repo: str, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def tree_state(repo: str) -> tuple[str | None, bool | None]:
    """The tree's commit and whether its tracked files match it (None: git cannot say)."""
    head = (_git(repo, "rev-parse", "--verify", "HEAD") or "").strip() or None
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return head, (None if status is None else not status.strip())


# ------------------------------------------------------------------ session
def _outcome(previous: str | None, report) -> str:
    """A test's outcome so far, given one more of its phase reports."""
    if getattr(report, "wasxfail", None) is not None:
        now = "xfailed" if report.skipped else "xpassed"
    elif report.failed:
        now = "failed" if report.when == "call" else "error"
    elif report.skipped:
        now = "skipped"
    else:
        now = "passed" if report.when == "call" else None
    return now if SEVERITY.get(now, 0) > SEVERITY.get(previous, 0) else previous


def _message(report) -> str:
    crash = getattr(getattr(report, "longrepr", None), "reprcrash", None)
    text = getattr(crash, "message", None) or getattr(report, "longreprtext", "") or ""
    if report.skipped and isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        text = str(report.longrepr[2])
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    return (lines[0] if lines else "")[:400]


def _plugin(record: dict, repo: str, selected: set[str] | None):
    import pytest

    tests: dict[str, dict] = record["tests"]
    ids: dict[str, str] = {}
    root: list[str] = [os.getcwd()]

    def repo_id(path, nodeid: str) -> str:
        """``nodeid`` with its file named relative to the tree."""
        rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(repo))
        rel = nodeid.split("::", 1)[0] if rel.startswith("..") else rel.replace(os.sep, "/")
        name = nodeid.split("::", 1)[1] if "::" in nodeid else ""
        return f"{rel}::{name}" if name else rel

    class RecorderPlugin:
        def pytest_configure(self, config):
            root[0] = str(config.rootpath)

        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, config, items):
            for item in items:
                ids[item.nodeid] = repo_id(item.path, item.nodeid)
                file, _, name = ids[item.nodeid].partition("::")
                record["collected"].setdefault(file, []).append(name)
            if selected is None:
                return
            files = {s for s in selected if "::" not in s}
            keep, drop = [], []
            for item in items:
                rid = ids[item.nodeid]
                (keep if rid in selected or rid.partition("::")[0] in files else drop).append(item)
            if drop:
                config.hook.pytest_deselected(items=drop)
                items[:] = keep

        def pytest_collectreport(self, report):
            if report.failed:
                file = report.nodeid.split("::", 1)[0]
                rid = repo_id(os.path.join(root[0], file), report.nodeid) if file else "<session>"
                tests.setdefault(
                    rid, {"outcome": "error", "duration": 0.0, "message": _message(report)}
                )

        def pytest_runtest_logstart(self, nodeid, location):
            tests[ids.get(nodeid, nodeid)] = {"outcome": None, "duration": 0.0, "message": ""}

        def pytest_runtest_logreport(self, report):
            entry = tests.get(ids.get(report.nodeid, report.nodeid))
            if entry is None:
                return
            before = entry["outcome"]
            entry["outcome"] = _outcome(before, report)
            entry["duration"] = round(entry["duration"] + float(report.duration or 0.0), 3)
            if entry["outcome"] != before and entry["outcome"] not in ("passed", None):
                entry["message"] = _message(report)

    return RecorderPlugin()


def _write(cache: str, record: dict) -> str:
    directory = os.path.join(cache, SESSIONS_DIRNAME)
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = os.path.join(directory, f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}.json")
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=None, separators=(",", ":"))
    os.replace(partial, path)
    return path


def _read_selection(path: str | None, items: list[str]) -> list[str]:
    lines = list(items)
    if path:
        with open(path, encoding="utf-8") as handle:
            lines += handle.read().splitlines()
    return [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]


def session(args: argparse.Namespace) -> int:
    """One pytest process, recorded: how each test it ran ended."""
    repo = os.path.abspath(args.repo)
    commit, clean = tree_state(repo)
    env = environment()
    record = {
        "version": RECORD_VERSION,
        "commit": commit,
        "clean": clean,
        "python": sys.executable,
        "argv": list(args.pytest_args),
        "started": _now(),
        "env": env,
        "env_digest": env_digest(env),
        "tests": {},
        "collected": {},
    }
    selected = set(_read_selection(args.select, [])) if args.select else None
    # As `python -m pytest` from the tree: the tree first on sys.path, this
    # script's directory nowhere on it.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, repo)
    os.chdir(repo)
    import pytest

    code = int(pytest.main(list(args.pytest_args), plugins=[_plugin(record, repo, selected)]))
    record["exit_code"] = code
    record["ended"] = _now()
    path = _write(args.cache, record)
    print(
        f"testcache: recorded {len(record['tests'])} test(s) at {commit or 'an unknown commit'} -> {path}"
    )
    return code


def _group(selection: list[str]) -> dict[str, list[str]]:
    """Selection lines by test file, in order; a bare file selects the whole file."""
    groups: dict[str, list[str]] = {}
    for item in selection:
        file = item.split("::", 1)[0]
        groups.setdefault(file, [])
        if item not in groups[file]:
            groups[file].append(item)
    return groups


def run(args: argparse.Namespace) -> int:
    """One recorded session per test file of the selection, in order."""
    repo = os.path.abspath(args.repo)
    worst = 0
    for file, items in _group(_read_selection(args.selection, args.items)).items():
        whole = file in items
        select = None
        command = [sys.executable, os.path.abspath(__file__), "session", "--cache", args.cache]
        command += ["--repo", repo]
        if not whole:
            select = os.path.join(args.cache, f".select-{os.getpid()}-{uuid.uuid4().hex[:6]}")
            os.makedirs(args.cache, exist_ok=True)
            with open(select, "w", encoding="utf-8") as handle:
                handle.write("\n".join(items) + "\n")
            command += ["--select", select]
        command += ["--", *args.pytest_args, file]
        print(
            f"testcache: {file}: {'whole file' if whole else f'{len(items)} test(s)'}", flush=True
        )
        started = time.monotonic()
        try:
            code = subprocess.run(command, cwd=repo, check=False).returncode
        finally:
            if select:
                try:
                    os.unlink(select)
                except OSError:
                    pass
        print(
            f"testcache: {file}: pytest exit {code} in {time.monotonic() - started:.0f}s",
            flush=True,
        )
        # 0 all passed, 1 some failed, 5 nothing collected: recorded either way.
        if code not in (0, 1, 5):
            worst = 2
    return worst


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="test_driver.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "session"):
        p = sub.add_parser(name)
        p.add_argument("--cache", required=True, help="The test cache directory.")
        p.add_argument("--repo", default=".", help="The tree under test (default: here).")
        if name == "run":
            p.add_argument("--selection", default="", help="A file of test files and node ids.")
            p.add_argument("items", nargs="*", help="Test files and node ids.")
        else:
            p.add_argument("--select", default="", help="Keep only these node ids.")
    if "--" in argv:
        at = argv.index("--")
        args = parser.parse_args(argv[:at])
        args.pytest_args = argv[at + 1 :]
    else:
        args = parser.parse_args(argv)
        args.pytest_args = []
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--environment"]:
        print("MV2FLOW_TEST_ENV " + env_digest(environment()))
        return 0
    args = _parse(argv)
    args.cache = os.path.abspath(args.cache)  # a session runs in the tree
    if args.action == "session":
        return session(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
