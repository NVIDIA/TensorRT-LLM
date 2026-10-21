# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native rebuilds that do not crash the processes holding the old libraries.

A rebuild that rewrites ``tensorrt_llm/libs/*.so`` in place makes a process on
another session that has them mapped take SIGSEGV, taking its whole
allocation with it. These tests pin the two measures: libraries get a fresh
inode before the build, and the rebuild is announced on every session so no
GPU job against that checkout starts mid-build.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import SCRIPTS_DIR, rebuild
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
)


def _scheduler(spool) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    return Scheduler(spool, SchedulerConfig(devices=("0", "1")), environ=env)


def _submit(spool, name, header):
    path = SpoolPaths(spool).pending / f"{name}.sh"
    path.write_text(f"#!/bin/bash\n#spool: {header}\ntrue\n", encoding="utf-8")


def test_a_fresh_inode_leaves_a_mapped_library_untouched(tmp_path):
    lib = tmp_path / "libs" / "libth_common.so"
    lib.parent.mkdir()
    lib.write_bytes(b"old-library-bytes")
    os.chmod(lib, 0o755)
    held = lib.open("rb")  # a running process's mapping
    before = os.stat(lib).st_ino
    try:
        refreshed = rebuild.refresh_inodes(tmp_path)
        assert refreshed == [lib]
        assert os.stat(lib).st_ino != before
        assert lib.read_bytes() == b"old-library-bytes"
        assert os.stat(lib).st_mode & 0o777 == 0o755
        # The build then rewrites the file in place -- the new inode, not the held one.
        with lib.open("r+b") as handle:
            handle.write(b"NEW")
        held.seek(0)
        assert held.read() == b"old-library-bytes"
    finally:
        held.close()
    assert not list(lib.parent.glob(".*.tmp"))


def test_symlinks_and_non_libraries_are_left_alone(tmp_path):
    (tmp_path / "libreal.so.1").write_bytes(b"x")
    (tmp_path / "libreal.so").symlink_to("libreal.so.1")
    (tmp_path / "notes.txt").write_text("x")
    inode = os.stat(tmp_path / "notes.txt").st_ino
    assert rebuild.refresh_inodes(tmp_path) == [tmp_path / "libreal.so.1"]
    assert (tmp_path / "libreal.so").is_symlink()
    assert os.stat(tmp_path / "notes.txt").st_ino == inode


def test_an_announcement_reaches_every_spool_and_is_retracted(tmp_path):
    spools = [tmp_path / "a", tmp_path / "b"]
    token = rebuild.announce(spools, tmp_path / "repo", note="test")
    for spool in spools:
        (marker,) = rebuild.active_announcements(spool)
        assert marker["token"] == token and marker["note"] == "test"
    rebuild.retract(spools, token)
    assert all(not rebuild.active_announcements(s) for s in spools)


def test_a_dead_builds_announcement_expires(tmp_path):
    rebuild.announce([tmp_path], tmp_path, max_age_s=10)
    assert rebuild.active_announcements(tmp_path)
    assert not rebuild.active_announcements(tmp_path, now=time.time() + 11)


def test_gpu_jobs_against_the_rebuilt_tree_wait_for_the_build(tmp_path):
    spool, repo, other = tmp_path / "spool", tmp_path / "repo", tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    sched = _scheduler(spool)
    token = rebuild.announce([spool], repo)
    _submit(spool, "same-tree", f"gpus=1 tree={repo}")
    _submit(spool, "no-tree", "gpus=1")
    _submit(spool, "other-tree", f"gpus=1 tree={other}")
    _submit(spool, "free-orch", "gpus=0")
    sched.tick()
    started = set(sched.jobs)
    assert started == {"other-tree", "free-orch"}
    reasons = {p["name"]: p["reason"] for p in sched.snapshot(time.time())["pending"]}
    assert "native rebuild of" in reasons["same-tree"] and "native rebuild of" in reasons["no-tree"]

    rebuild.retract([spool], token)
    end = time.monotonic() + 20
    receipts = [SpoolPaths(spool).receipt(name) for name in ("same-tree", "no-tree")]
    while time.monotonic() < end and not all(path.exists() for path in receipts):
        sched.tick()
        time.sleep(0.02)
    assert SpoolPaths(spool).receipt("same-tree").exists()
    assert SpoolPaths(spool).receipt("no-tree").exists()


def _checkout(root: Path, *, own_build: bool = True) -> Path:
    """A tree with a ``tensorrt_llm`` package; ``own_build`` gives it a native build of its own."""
    (root / "tensorrt_llm").mkdir(parents=True)
    (root / "tensorrt_llm" / "__init__.py").write_text("")
    if own_build:
        (root / "tensorrt_llm" / "libs").mkdir()
        (root / "tensorrt_llm" / "libs" / "libth_common.so").write_bytes(b"x")
        (root / "tensorrt_llm" / "bindings.so").write_bytes(b"x")
    return root


def _linked(root: Path, repo: Path) -> Path:
    """A snapshot or worktree: its own sources, the checkout's native build by symlink."""
    tree = _checkout(root, own_build=False)
    for entry in ("libs", "bindings.so"):
        (tree / "tensorrt_llm" / entry).symlink_to(repo / "tensorrt_llm" / entry)
    return tree


def test_a_linked_trees_native_build_is_where_its_links_resolve(tmp_path):
    repo = _checkout(tmp_path / "repo")
    snapshot = _linked(tmp_path / "snapshots" / "abc", repo)
    found = rebuild.native_locations(snapshot)
    assert str((repo / "tensorrt_llm" / "libs").resolve()) in found
    assert str((repo / "tensorrt_llm" / "bindings.so").resolve()) in found
    assert rebuild.native_locations(repo) == [str((repo / "tensorrt_llm").resolve())]
    assert rebuild.native_locations(tmp_path / "nothing") == [
        str((tmp_path / "nothing" / "tensorrt_llm").resolve())
    ]


def test_an_announcement_records_where_the_rebuilt_build_is(tmp_path):
    repo = _checkout(tmp_path / "repo")
    rebuild.announce([tmp_path / "spool"], repo)
    (marker,) = rebuild.active_announcements(tmp_path / "spool")
    assert marker["native"] == rebuild.native_locations(repo)


def test_jobs_whose_native_build_links_into_the_rebuilt_tree_wait_too(tmp_path):
    """A snapshot or worktree loads the checkout's libraries; a rebuild there rewrites them."""
    spool = tmp_path / "spool"
    repo = _checkout(tmp_path / "repo")
    snapshot = _linked(tmp_path / "ws" / "snapshots" / "abc", repo)
    worktree = _linked(tmp_path / "ws" / "tasks" / "t1", repo)
    unrelated = _checkout(tmp_path / "ws" / "unrelated")
    package_link = tmp_path / "ws" / "package-link"
    package_link.mkdir(parents=True)
    (package_link / "tensorrt_llm").symlink_to(repo / "tensorrt_llm")
    # A worktree under the checkout with a build of its own shares nothing with it.
    own = _checkout(repo / ".worktrees" / "own")
    sched = _scheduler(spool)
    token = rebuild.announce([spool], repo)
    for name, tree in (
        ("snapshot", snapshot),
        ("worktree", worktree),
        ("package-link", package_link),
        ("own-build", own),
        ("unrelated", unrelated),
    ):
        _submit(spool, name, f"gpus=1 tree={tree}")
    sched.tick()
    assert set(sched.jobs) == {"own-build", "unrelated"}
    reasons = {p["name"]: p["reason"] for p in sched.snapshot(time.time())["pending"]}
    for name, tree in (("snapshot", snapshot), ("worktree", worktree)):
        assert reasons[name].startswith(f"native rebuild of {repo} in progress")
        assert f"{tree} shares its native build" in reasons[name]
    assert "native rebuild of" in reasons["package-link"]

    rebuild.retract([spool], token)
    end = time.monotonic() + 20
    receipts = [SpoolPaths(spool).receipt(n) for n in ("snapshot", "worktree", "package-link")]
    while time.monotonic() < end and not all(path.exists() for path in receipts):
        sched.tick()
        time.sleep(0.02)
    assert all(path.exists() for path in receipts)


def test_a_rebuilt_worktree_that_links_the_checkouts_build_holds_the_checkouts_jobs(tmp_path):
    """Its build writes through the links, into the libraries the checkout's jobs load."""
    spool = tmp_path / "spool"
    repo = _checkout(tmp_path / "repo")
    worktree = _linked(tmp_path / "ws" / "tasks" / "t1", repo)
    other = _checkout(tmp_path / "other")
    sched = _scheduler(spool)
    rebuild.announce([spool], worktree)
    _submit(spool, "checkout", f"gpus=1 tree={repo}")
    _submit(spool, "other", f"gpus=1 tree={other}")
    sched.tick()
    assert set(sched.jobs) == {"other"}
    waiting = {p["name"]: p["reason"] for p in sched.snapshot(time.time())["pending"]}
    assert list(waiting) == ["checkout"]
    assert f"{repo} shares its native build" in waiting["checkout"]


def test_a_marker_without_its_locations_has_them_resolved(tmp_path):
    """An announcement from before the locations were recorded still holds linked trees."""
    repo = _checkout(tmp_path / "repo")
    snapshot = _linked(tmp_path / "snapshots" / "abc", repo)
    other = _checkout(tmp_path / "other")
    marker = {"tree": str(repo), "started": time.time()}
    assert rebuild.loads_rebuilt(str(snapshot), marker)
    assert rebuild.loads_rebuilt(str(repo), marker)
    assert not rebuild.loads_rebuilt(str(other), marker)
    assert not rebuild.loads_rebuilt("", marker)


@pytest.fixture
def fake_checkout(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tensorrt_llm" / "libs").mkdir(parents=True)
    (repo / "tensorrt_llm" / "__init__.py").write_text("__version__ = '0.0.test'\n")
    (repo / "tensorrt_llm" / "libs" / "libth_common.so").write_bytes(b"old")
    # A stand-in for build_wheel.py: records what the spools said during the
    # build, then rewrites the library IN PLACE, as the real one does.
    (repo / "scripts" / "build_wheel.py").write_text(
        "import json, os, sys\n"
        "spool = os.environ['TEST_SPOOL']\n"
        "markers = sorted(os.listdir(os.path.join(spool, 'rebuild')))\n"
        "open('build-saw.json', 'w').write(json.dumps({'argv': sys.argv[1:], 'markers': markers}))\n"
        "with open('tensorrt_llm/libs/libth_common.so', 'r+b') as f:\n"
        "    f.write(b'new')\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        check=True,
    )
    return repo


def test_incremental_build_announces_refreshes_builds_and_retracts(tmp_path, fake_checkout):
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    lib = fake_checkout / "tensorrt_llm" / "libs" / "libth_common.so"
    held = lib.open("rb")
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("MV2FLOW_", "PYTHON"))},
        "TEST_SPOOL": str(spool),
        "MV2FLOW_CUDA_ARCHS": "100-real",
    }
    try:
        done = subprocess.run(
            [
                "bash",
                str(SCRIPTS_DIR / "incremental_build.sh"),
                "--repo",
                str(fake_checkout),
                "--spool",
                str(spool),
                "--log-dir",
                str(tmp_path / "logs"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert held.read() == b"old", "the running process's library must not change under it"
    finally:
        held.close()
    assert lib.read_bytes() == b"new"
    saw = json.loads((fake_checkout / "build-saw.json").read_text())
    assert len(saw["markers"]) == 1, "the build ran while the rebuild was announced"
    assert saw["argv"][:2] == ["--use_ccache", "--skip_building_wheel"]
    assert "--cuda_architectures" in saw["argv"] and "100-real" in saw["argv"]
    assert not rebuild.active_announcements(spool), "the announcement is retracted afterwards"
    assert "import ok 0.0.test" in done.stdout
    assert "fresh inodes for 1 shared librar" in done.stdout
    (log,) = (tmp_path / "logs").glob("incremental-*.log")
    assert "INCREMENTAL_BUILD_WALL_SECONDS" in log.read_text()


def test_a_failed_build_still_retracts_its_announcement(tmp_path, fake_checkout):
    (fake_checkout / "scripts" / "build_wheel.py").write_text("raise SystemExit(5)\n")
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MV2FLOW_", "PYTHON"))}
    done = subprocess.run(
        [
            "bash",
            str(SCRIPTS_DIR / "incremental_build.sh"),
            "--repo",
            str(fake_checkout),
            "--spool",
            str(spool),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 5
    assert not rebuild.active_announcements(spool)
    assert "import ok" not in done.stdout


def test_the_build_script_names_no_machine_specific_path():
    text = (SCRIPTS_DIR / "incremental_build.sh").read_text()
    for forbidden in ("/scratch/", "/lustre/", "/home/"):
        assert forbidden not in text
    assert Path(SCRIPTS_DIR / "incremental_build.sh").stat().st_mode & 0o111


def _installs_a_package(repo: Path) -> None:
    """A build_wheel.py that pip-installs into the venv it runs in, as the real one does."""
    (repo / "scripts" / "build_wheel.py").write_text(
        "import os, sysconfig\n"
        "os.makedirs(os.path.join(sysconfig.get_paths()['purelib'], 'drift-9.9.dist-info'))\n"
    )


def _build(tmp_path: Path, repo: Path, run_python: Path, **env: str):
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    return subprocess.run(
        ["bash", str(SCRIPTS_DIR / "incremental_build.sh"), "--repo", str(repo), "--spool",
         str(spool), "--log-dir", str(tmp_path / "logs"), "--no-import-check"],
        env={
            **{k: v for k, v in os.environ.items() if not k.startswith(("MV2FLOW_", "PYTHON"))},
            "MV2FLOW_BUILD_PYTHON": str(run_python),
            **env,
        },
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip


def test_a_build_installs_into_a_venv_of_its_own_and_never_into_the_runs(tmp_path, fake_checkout):
    """Two native builds both pip-installed into the run's venv, and the second then failed."""
    run_venv = tmp_path / "run-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(run_venv)], check=True)
    run_python = run_venv / "bin" / "python"
    purelib = subprocess.run(
        [str(run_python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    _installs_a_package(fake_checkout)

    done = _build(tmp_path, fake_checkout, run_python)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (Path(purelib) / "drift-9.9.dist-info").exists(), "the run's venv is untouched"
    build_venv = fake_checkout / ".dev-build" / "build-venv"
    assert list(build_venv.glob("lib/python*/site-packages/drift-9.9.dist-info"))
    assert list(build_venv.glob("lib/python*/site-packages/mv2flow-run-venv.pth"))

    unguarded = _build(tmp_path, fake_checkout, run_python, MV2FLOW_BUILD_ISOLATED="0")
    assert unguarded.returncode == 65
    assert "the run's venv" in unguarded.stdout + unguarded.stderr
