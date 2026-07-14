#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""End-to-end tests for ``3rdparty/fetch_cache.py update``.

Hermetic — every fixture is a tiny git repo built locally, so the suite
runs in CI with no network and no GitHub access.  Each cycle ends by
cloning with ``git clone --reference <bare>`` and checking that the
consumer holds **zero git objects of its own**: that is what proves the
reference mechanism actually carried the bytes.

Coverage:

* top-level dep, full and shallow modes — initial update, then a fresh
  src at a new commit re-updates the same bare; both consumers end up
  object-less;
* parent dep with a nested submodule — bare materializes for parent and
  submodule, both consumer git dirs are object-less;
* LFS-tracked binary — the oid lands in the bare's LFS pool with bytes
  matching the source.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FETCH_CACHE = REPO_ROOT / "3rdparty" / "fetch_cache.py"

# Hermetic git identity so the test doesn't read the host's git config.
_GIT_IDENT = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env() -> dict[str, str]:
    e = os.environ.copy()
    e.update(_GIT_IDENT)
    return e


def _run(cmd, *, cwd=None, check=True):
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=_env(),
        check=check,
        capture_output=True,
        text=True,
    )


def _git(*args, cwd=None, check=True):
    """Plain git with `protocol.file.allow=always`.

    Permits fixture clones from `file://` upstreams (top-level and
    submodule) regardless of the host's default `protocol.allow` policy.
    """
    return _run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        check=check,
    )


def _file_url(path: Path) -> str:
    """``file://`` URL for *path*.

    Forces ``git clone`` to use the smart protocol instead of git's
    local-clone optimization (hardlink/copy of objects, no haves
    advertised).  Production cmake clones from `https://github.com/...`
    URLs which always use the smart protocol; the test fixture lives
    on local disk, so this prefix is what makes ``--reference``
    actually carry the load via alternates.
    """
    return f"file://{path.resolve()}"


def _have_lfs() -> bool:
    try:
        return subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _fetch_cache_update(cache_dir: Path, src: Path) -> None:
    r = _run(
        [sys.executable, FETCH_CACHE, "update", "--cache-dir", cache_dir, "--src", src],
    )
    assert r.returncode == 0, f"fetch_cache.py failed:\n{r.stdout}\n{r.stderr}"


def _local_object_counts(repo: Path) -> tuple[int, int]:
    """Return ``(loose, in_pack)`` counts excluding alternates.

    A `clone --reference <bare>` that fully covered upstream history
    produces ``(0, 0)`` — the bare carried every object via alternates,
    the consumer holds none of its own.
    """
    r = _git("-C", repo, "count-objects", "-v")
    loose, in_pack = 0, 0
    for line in r.stdout.splitlines():
        key, _, val = line.partition(":")
        if key == "count":
            loose = int(val.strip())
        elif key == "in-pack":
            in_pack = int(val.strip())
    return loose, in_pack


def _alternates_target(gitdir: Path) -> str | None:
    alt = gitdir / "objects" / "info" / "alternates"
    if not alt.is_file():
        return None
    return alt.read_text().strip()


def _init_bare(path: Path) -> Path:
    _git("init", "--bare", "--initial-branch=main", path)
    return path


def _init_worktree(path: Path, origin_url: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "--initial-branch=main", path)
    _git("remote", "add", "origin", origin_url, cwd=path)
    return path


def _commit(worktree: Path, files: dict[str, bytes], msg: str) -> None:
    for name, content in files.items():
        p = worktree / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    _git("add", "-A", cwd=worktree)
    _git("commit", "-m", msg, cwd=worktree)


def _set_origin_url(worktree: Path, url: str) -> None:
    """Re-point a freshly-cloned worktree's origin at a stable URL.

    The test's upstream lives at a tempdir path, but the cache key
    `_repo_name(remote.origin.url)` should be deterministic across
    runs.
    """
    _git("remote", "set-url", "origin", url, cwd=worktree)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Per-test workspace — pytest auto-cleans tmp_path."""
    (tmp_path / "cache").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# top-level dep — two cycles, full and shallow
# ---------------------------------------------------------------------------


def _run_top_level_cycles(workspace: Path, *, shallow: bool) -> None:
    cache = workspace / "cache"
    upstream = _init_bare(workspace / "upstream.git")
    origin_url = "https://example.com/sample.git"
    sub = "shallow" if shallow else "full"
    bare = cache / sub / "sample.git"

    # Seed worktree drives upstream's history (bare repos can't take
    # commits directly).
    seed = _init_worktree(workspace / "seed", str(upstream))
    _commit(seed, {"a.txt": b"first\n"}, "c1")
    _git("push", "origin", "main", cwd=seed)

    depth = ["--depth", "1"] if shallow else []
    upstream_url = _file_url(upstream)

    # Cycle 1: clone seed→src1, update cache, fresh consumer must be
    # object-less.  All clones from upstream use file:// so --depth
    # actually produces a shallow clone (local-path mode ignores
    # --depth) and so --reference clones go through the smart protocol.
    src1 = workspace / "src1"
    _git("clone", *depth, upstream_url, src1)
    _set_origin_url(src1, origin_url)
    _fetch_cache_update(cache, src1)

    assert (bare / "HEAD").is_file(), f"bare not created at {bare}"

    consumer1 = workspace / "consumer1"
    _git("clone", *depth, "--reference", bare, upstream_url, consumer1)
    assert _alternates_target(consumer1 / ".git") == str((bare / "objects").resolve())
    assert _local_object_counts(consumer1) == (0, 0), (
        "cycle 1: consumer carries own objects — reference mechanism didn't"
        " carry the upstream content"
    )

    # Cycle 2: new commit upstream → fresh src2 → re-update same bare →
    # fresh consumer at the new tip must still be object-less.
    _commit(seed, {"a.txt": b"second\n"}, "c2")
    _git("push", "origin", "main", cwd=seed)

    src2 = workspace / "src2"
    _git("clone", *depth, upstream_url, src2)
    _set_origin_url(src2, origin_url)
    _fetch_cache_update(cache, src2)

    consumer2 = workspace / "consumer2"
    _git("clone", *depth, "--reference", bare, upstream_url, consumer2)
    assert _local_object_counts(consumer2) == (0, 0), (
        "cycle 2: incremental update didn't carry the new commit's objects into the bare"
    )
    assert (consumer2 / "a.txt").read_bytes() == b"second\n"


def test_top_level_full(workspace: Path) -> None:
    _run_top_level_cycles(workspace, shallow=False)


def test_top_level_shallow(workspace: Path) -> None:
    _run_top_level_cycles(workspace, shallow=True)


# ---------------------------------------------------------------------------
# submodule — bare materializes for both parent and submodule
# ---------------------------------------------------------------------------


def test_submodule(workspace: Path) -> None:
    cache = workspace / "cache"
    parent_upstream = _init_bare(workspace / "parent.git")
    sub_upstream = _init_bare(workspace / "sub.git")

    # Seed the submodule first so the parent's `submodule add` finds a
    # populated remote.
    sub_seed = _init_worktree(workspace / "sub_seed", str(sub_upstream))
    _commit(sub_seed, {"sub.txt": b"sub-v1\n"}, "s1")
    _git("push", "origin", "main", cwd=sub_seed)

    par_seed = _init_worktree(workspace / "par_seed", str(parent_upstream))
    _commit(par_seed, {"top.txt": b"top-v1\n"}, "p1")
    # `submodule add` with a file:// URL: the URL ends up in
    # `.gitmodules`, and the consumer's submodule init below needs the
    # smart protocol (not git's local-clone optimization) so its
    # --reference alternates actually carry the load.
    _git("submodule", "add", _file_url(sub_upstream), "lib/sub", cwd=par_seed)
    _git("commit", "-m", "add submodule", cwd=par_seed)
    _git("push", "origin", "main", cwd=par_seed)

    # Mimic cmake FetchContent populate: clone parent with submodule init.
    src = workspace / "src"
    _git("clone", "--recurse-submodules", parent_upstream, src)
    _set_origin_url(src, "https://example.com/parent.git")

    _fetch_cache_update(cache, src)

    # fetch_cache.py walks <src>/.git/modules/, so the submodule's bare
    # also lands in the cache.  Cache key for the submodule comes from
    # its own remote.origin.url, which `submodule add` set to the local
    # file path — basename is `sub`.
    parent_bare = cache / "full" / "parent.git"
    sub_bare = cache / "full" / "sub.git"
    assert (parent_bare / "HEAD").is_file(), f"missing {parent_bare}"
    assert (sub_bare / "HEAD").is_file(), f"missing {sub_bare}"

    # Consumer: parent clones with parent_bare reference; submodule init
    # uses sub_bare.  Both git dirs must be object-less.
    consumer = workspace / "consumer"
    _git("clone", "--reference", parent_bare, _file_url(parent_upstream), consumer)
    _git("submodule", "update", "--init", "--reference", sub_bare, cwd=consumer)

    assert _local_object_counts(consumer) == (0, 0), "parent consumer carries own objects"

    # Submodule's gitdir is <consumer>/.git/modules/lib/sub.
    submod_gitdir = consumer / ".git" / "modules" / "lib" / "sub"
    assert (submod_gitdir / "HEAD").is_file()
    assert _alternates_target(submod_gitdir) == str((sub_bare / "objects").resolve())
    assert _local_object_counts(consumer / "lib" / "sub") == (0, 0), (
        "submodule consumer carries own objects"
    )


# ---------------------------------------------------------------------------
# LFS — oid lands in the bare's pool
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_lfs(), reason="git-lfs not installed")
def test_lfs_pool(workspace: Path) -> None:
    cache = workspace / "cache"
    src = workspace / "src"
    src.mkdir()

    # No upstream needed: fetch_cache.py reads LFS objects from the
    # src worktree directly.  Cache key comes from remote.origin.url.
    _git("init", "--initial-branch=main", src)
    _git("remote", "add", "origin", "https://example.com/with-lfs.git", cwd=src)
    _git("lfs", "install", "--local", cwd=src)
    _git("lfs", "track", "*.bin", cwd=src)

    payload = b"A" * 4096 + b"\n"
    oid = hashlib.sha256(payload).hexdigest()
    (src / "a.bin").write_bytes(payload)
    _git("add", "-A", cwd=src)
    _git("commit", "-m", "add lfs blob", cwd=src)

    # Clean filter places oid under src/.git/lfs/objects/.
    src_lfs_path = src / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    assert src_lfs_path.is_file(), f"LFS clean filter didn't write {src_lfs_path}"
    assert src_lfs_path.read_bytes() == payload

    _fetch_cache_update(cache, src)

    bare_lfs_path = cache / "full" / "with-lfs.git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    assert bare_lfs_path.is_file(), f"fetch_cache.py didn't ingest oid into {bare_lfs_path}"
    assert bare_lfs_path.read_bytes() == payload, "ingested LFS bytes differ from src"

    # Re-run: idempotent (existing oid skipped, no error).
    bare_inode = bare_lfs_path.stat().st_ino
    _fetch_cache_update(cache, src)
    assert bare_lfs_path.is_file()
    assert bare_lfs_path.stat().st_ino == bare_inode, (
        "second update unexpectedly rewrote the existing LFS object"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
