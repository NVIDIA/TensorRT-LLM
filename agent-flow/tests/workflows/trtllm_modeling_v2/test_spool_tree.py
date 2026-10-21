# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--tree``: a job runs against one checkout, and a result that mixed checkouts is refused.

An MPI worker resolves the engine package from its working directory ahead
of ``PYTHONPATH``, so a worktree run with only ``PYTHONPATH`` set runs the
worktree on rank 0 and the main checkout everywhere else. These tests build a
fake package in two checkouts and reproduce that with plain ``python3``
processes, one standing in for each rank.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    CommandSpec,
    select_command_executor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import provenance, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_MIXED_CHECKOUT,
    RC_REFUSED,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)

IMPORT = "python3 -c 'import fakepkg'"


@pytest.fixture
def checkouts(tmp_path):
    trees = {}
    for name in ("main", "wt"):
        package = tmp_path / name / "fakepkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"WHERE = {name!r}\n", encoding="utf-8")
        trees[name] = tmp_path / name
    return trees


@pytest.fixture
def spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def scheduler(spool, environ=None) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PYTHON"))}
    env.update(environ or {})
    return Scheduler(spool, SchedulerConfig(devices=("0",), leftover_grace_s=0), environ=env)


def run_job(spool, name, body, header, environ=None) -> dict:
    paths = SpoolPaths(spool)
    (paths.pending / f"{name}.sh").write_text(
        f"#!/bin/bash\n#spool: {header}\n{body}\n", encoding="utf-8"
    )
    sched = scheduler(spool, environ)
    end = time.monotonic() + 30
    while time.monotonic() < end:
        sched.tick()
        if paths.receipt(name).exists() and not sched.jobs:
            return json.loads(paths.record(name).read_text())
        time.sleep(0.02)
    raise AssertionError(paths.log(name).read_text() if paths.log(name).exists() else "no log")


def test_a_tree_job_runs_in_the_tree_and_imports_from_it(spool, checkouts):
    body = f"pwd; echo PYTHONPATH=$PYTHONPATH; {IMPORT}"
    record = run_job(spool, "t", body, f"gpus=1 tree={checkouts['wt']} resolve=fakepkg")
    assert record["rc"] == 0 and record["reason"] == "exit"
    log = SpoolPaths(spool).log("t").read_text()
    assert f"\n{checkouts['wt']}\n" in log
    assert "=== tree " in log
    assert f"module=fakepkg origin={checkouts['wt']}/fakepkg/__init__.py" in log
    entries = provenance.read_records(SpoolPaths(spool).resolve_log("t"))
    assert [e["origin"] for e in entries] == [f"{checkouts['wt']}/fakepkg/__init__.py"]


def test_ranks_that_resolved_different_checkouts_are_refused(spool, checkouts):
    """The mixed shape: one rank imports the tree, another the main checkout via its cwd."""
    body = f"{IMPORT}; (cd {checkouts['main']} && {IMPORT})"
    record = run_job(spool, "mixed", body, f"gpus=1 tree={checkouts['wt']} resolve=fakepkg")
    assert record["rc"] == RC_MIXED_CHECKOUT
    assert record["reason"] == "mixed_checkout"
    assert "2 checkouts" in record["detail"]
    assert str(checkouts["main"]) in record["detail"] and str(checkouts["wt"]) in record["detail"]


def test_a_failed_mixed_job_keeps_its_exit_code_but_says_why(spool, checkouts):
    body = f"{IMPORT}; (cd {checkouts['main']} && {IMPORT}); exit 3"
    record = run_job(spool, "mixedfail", body, f"gpus=1 tree={checkouts['wt']} resolve=fakepkg")
    assert record["rc"] == 3 and record["reason"] == "mixed_checkout"


def test_every_process_in_the_wrong_checkout_is_refused_too(spool, checkouts):
    body = f"cd {checkouts['main']} && {IMPORT}"
    record = run_job(spool, "foreign", body, f"gpus=1 tree={checkouts['wt']} resolve=fakepkg")
    assert record["rc"] == RC_MIXED_CHECKOUT
    assert "not to the job's tree" in record["detail"]


def test_without_a_tree_disagreeing_processes_are_still_caught(spool, checkouts):
    body = f"(cd {checkouts['main']} && {IMPORT}); (cd {checkouts['wt']} && {IMPORT})"
    record = run_job(spool, "notree", body, "gpus=1 resolve=fakepkg")
    assert record["reason"] == "mixed_checkout"


def test_a_job_that_never_imports_the_package_is_untouched(spool, checkouts):
    record = run_job(spool, "plain", "echo hi", f"gpus=1 tree={checkouts['wt']}")
    assert record["rc"] == 0 and record["reason"] == "exit"


def test_the_hook_can_be_turned_off(spool, checkouts):
    body = f"echo PP=$PYTHONPATH; {IMPORT}"
    record = run_job(spool, "off", body, f"gpus=1 tree={checkouts['wt']} resolve=off")
    assert record["rc"] == 0
    assert not SpoolPaths(spool).resolve_log("off").exists()
    assert "resolvehook" not in SpoolPaths(spool).log("off").read_text()


def test_the_hook_runs_the_sitecustomize_it_shadows(spool, checkouts, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text("print('SITE-HOOK-RAN')\n", encoding="utf-8")
    record = run_job(
        spool,
        "chain",
        IMPORT,
        "gpus=1 resolve=fakepkg",
        environ={"PYTHONPATH": f"{site}:{checkouts['wt']}"},
    )
    assert record["rc"] == 0
    log = SpoolPaths(spool).log("chain").read_text()
    assert "SITE-HOOK-RAN" in log and "[mv2flow-resolve]" in log


def test_a_tree_the_watcher_cannot_see_is_refused(spool, tmp_path):
    record = run_job(spool, "ghost", "true", f"gpus=1 tree={tmp_path / 'nope'}")
    assert record["rc"] == RC_REFUSED and "not a directory" in record["detail"]


def test_checkout_roots():
    assert provenance.checkout_root("tensorrt_llm", "/a/b/tensorrt_llm/__init__.py") == "/a/b"
    assert provenance.checkout_root("x.y", "/r/x/y/__init__.py") == "/r"
    assert provenance.checkout_root("x.y", "/r/x/y.py") == "/r"
    assert provenance.checkout_root("ext", "/r/ext.cpython-312.so") == "/r"


def test_two_spellings_of_one_checkout_are_one_checkout(tmp_path, checkouts):
    """The same storage under two mount prefixes must not read as a mixed run."""
    alias = tmp_path / "alias"
    alias.symlink_to(checkouts["wt"])
    records = [
        {"module": "fakepkg", "origin": f"{checkouts['wt']}/fakepkg/__init__.py"},
        {"module": "fakepkg", "origin": f"{alias}/fakepkg/__init__.py"},
    ]
    assert provenance.check(records, tree=str(checkouts["wt"])).ok


def test_spoolctl_tree_sets_the_header_and_the_working_directory(spool, checkouts, monkeypatch):
    monkeypatch.setattr(
        spoolctl,
        "queue_job",
        lambda spool_, name, script, owner=None: captured.update(script=script),
    )
    captured: dict = {}
    assert (
        spoolctl.main(
            ["--spool", str(spool), "launch", "--tree", str(checkouts["wt"]), "free-x", "true"]
        )
        == 0
    )
    assert f"tree={checkouts['wt']}" in captured["script"]
    assert f"cd {checkouts['wt']} || exit 1" in captured["script"]


def test_the_orchestrators_gates_declare_the_checkout_under_test(tmp_path, checkouts):
    env = RunEnvironment(kind="session_spool", python="/p", spool=str(tmp_path))
    executor = select_command_executor(env, tree=checkouts["wt"])
    assert executor.tree == str(checkouts["wt"].resolve())

    executor.poll_s = 0.01
    executor.execute(CommandSpec(argv="true", timeout_s=0))
    script = next((tmp_path / "pending").glob("*.sh")).read_text()
    assert f"tree={checkouts['wt'].resolve()}" in script


def test_the_hook_ships_with_the_package():
    from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import RESOLVE_HOOK_DIR

    assert (RESOLVE_HOOK_DIR / "sitecustomize.py").is_file()
    assert Path(RESOLVE_HOOK_DIR).name == "resolvehook"
