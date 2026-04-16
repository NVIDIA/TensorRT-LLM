#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
FetchContent cache manager for TensorRT-LLM 3rdparty dependencies.

Creates and maintains bare git reference repos that accelerate
``git clone`` via the ``--reference`` mechanism.

No explicit ``init`` step — the cache is populated from local build
artifacts the first time ``update`` runs after a successful build.

Called automatically by build_wheel.py after cmake build.

Usage::

    python fetch_cache.py update --cache-dir <dir> --build-dir <dir>
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

# Prevents accidental GC of objects that clones reference via alternates.
SAFETY_CONFIG = {
    "gc.auto": "0",
    "gc.pruneExpire": "never",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _repo_name(url: str) -> str:
    """``https://x/y/Foo.git`` -> ``Foo``."""
    return os.path.basename(url.rstrip("/")).removesuffix(".git")


def _run_git(args, **kwargs):
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    return subprocess.run(["git"] + args, **kwargs)


def _apply_safety_config(bare_dir: str) -> None:
    for key, val in SAFETY_CONFIG.items():
        _run_git(["config", key, val], cwd=bare_dir)


def _get_origin_url(repo_dir: str) -> str | None:
    """Get the origin URL from a git repo (work tree or bare)."""
    r = _run_git(["config", "--get", "remote.origin.url"], cwd=repo_dir)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return None


def _ensure_cache(src_dir: str, cache_dir: str) -> str | None:
    """Create or update a bare cache repo from a local *src_dir*.

    Returns the cache path on success, None on failure.
    """
    url = _get_origin_url(src_dir)
    if not url:
        return None
    name = _repo_name(url)
    bare = os.path.join(cache_dir, f"{name}.git")

    if os.path.isfile(os.path.join(bare, "HEAD")):
        # Merge new objects into existing cache
        r = _run_git(
            ["fetch", "--no-tags", "--no-auto-gc",
             os.path.realpath(src_dir),
             "+refs/heads/*:refs/fetch-cache/heads/*",
             "+refs/tags/*:refs/fetch-cache/tags/*"],
            cwd=bare,
        )
        if r.returncode != 0:
            logger.info("  %s: merge failed, skipping", name)
        return bare

    # First time — create bare cache from local repo (no network)
    logger.info("  %s: creating cache from local repo", name)
    r = _run_git(["clone", "--bare", os.path.realpath(src_dir), bare])
    if r.returncode != 0:
        logger.info("  %s: clone --bare failed, skipping", name)
        if os.path.isdir(bare):
            shutil.rmtree(bare, ignore_errors=True)
        return None
    _apply_safety_config(bare)
    return bare


# ---------------------------------------------------------------------------
# update — create-or-merge cache from build _deps
# ---------------------------------------------------------------------------

def cmd_update(cache_dir: str, build_dir: str) -> int:
    """Scan ``_deps/`` for git repos, create or update bare caches."""
    deps_dir = os.path.join(build_dir, "_deps")
    if not os.path.isdir(deps_dir):
        logger.info("No _deps directory in %s, nothing to update", build_dir)
        return 0

    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Updating FetchContent cache: %s", cache_dir)

    for entry in sorted(os.listdir(deps_dir)):
        if not entry.endswith("-src"):
            continue
        src_dir = os.path.join(deps_dir, entry)
        git_marker = os.path.join(src_dir, ".git")
        if not (os.path.isdir(git_marker) or os.path.isfile(git_marker)):
            continue

        # Top-level dep
        result = _ensure_cache(src_dir, cache_dir)
        if result:
            logger.info("  %s: ok", entry)

        # Submodule repos (scan .git/modules/)
        _update_submodules(src_dir, cache_dir)

    return 0


def _update_submodules(src_dir: str, cache_dir: str) -> None:
    """Walk ``.git/modules/`` to find and cache submodule repos."""
    # Resolve the actual git dir (handles both .git/ dir and .git file)
    r = _run_git(["rev-parse", "--git-dir"], cwd=src_dir)
    if r.returncode != 0:
        return
    git_dir = os.path.join(src_dir, r.stdout.strip())
    modules_dir = os.path.join(git_dir, "modules")
    if not os.path.isdir(modules_dir):
        return

    for root, dirs, files in os.walk(modules_dir):
        # A real git dir has HEAD + objects/ (not just HEAD)
        if "HEAD" not in files or "objects" not in dirs:
            continue
        result = _ensure_cache(root, cache_dir)
        if result:
            rel = os.path.relpath(root, modules_dir)
            logger.info("    submodule %s: ok", rel)
        # Don't descend into this submodule's internals
        dirs[:] = [d for d in dirs if d == "modules"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="-- [fetch-cache] %(message)s",
    )

    p = argparse.ArgumentParser(description="FetchContent cache manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("update",
                        help="Create or update cache from build artifacts")
    sp.add_argument("--cache-dir", required=True, help="Cache directory")
    sp.add_argument("--build-dir", required=True, help="CMake build directory")

    args = p.parse_args()

    if args.cmd == "update":
        return cmd_update(args.cache_dir, args.build_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
