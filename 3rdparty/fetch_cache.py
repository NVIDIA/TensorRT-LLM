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

No explicit ``init`` step — the cache is populated on the fly, every
time ``FetchContent_MakeAvailable`` finishes populating a dep.

Two ``update`` modes:

* ``--src <path>`` single-module mode — invoked by the
  ``FetchContent_MakeAvailable`` override in ``3rdparty/CMakeLists.txt``
  the moment a dep's src is populated (and walks its ``.git/modules/``
  for submodules too).  This is the primary path.
* ``--build-dir <path>`` full-scan mode — walks ``_deps/`` and
  ``.git/modules/`` to rebuild the cache in bulk.  Used for manual
  repair or when the cmake override is not available (older trees).

Usage::

    python fetch_cache.py update --cache-dir <dir> --src <path>
    python fetch_cache.py update --cache-dir <dir> --build-dir <dir>
"""

import argparse
import logging
import os
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


def _remove_shallow(bare_dir: str) -> None:
    """Remove the ``shallow`` file so the repo can be used as --reference."""
    shallow = os.path.join(bare_dir, "shallow")
    try:
        if os.path.isfile(shallow):
            os.remove(shallow)
    except OSError:
        pass


def _ensure_cache(src_dir: str, cache_dir: str) -> str | None:
    """Create or update a bare cache repo from a local *src_dir*.

    Uses ``git init --bare`` + ``git fetch`` instead of ``git clone --bare``
    to avoid inheriting shallow state from the source repo.  A shallow bare
    repo cannot be used as ``--reference`` (git rejects it outright).

    Concurrency / partial-failure: every step is idempotent — ``git init
    --bare`` re-runs harmlessly on an existing bare, ``git config`` is a
    plain key/value write, and ``git fetch`` resumes whatever objects/refs
    are missing.  We deliberately do *not* delete a partial bare on
    failure: leaving it in place lets the next ``update`` heal it via the
    "existing cache" path and avoids a "rmtree races fetch" hazard
    between parallel builds sharing the same cache.

    Returns the cache path on success, None on failure.
    """
    url = _get_origin_url(src_dir)
    if not url:
        return None
    name = _repo_name(url)
    bare = os.path.join(cache_dir, f"{name}.git")
    real_src = os.path.realpath(src_dir)

    if os.path.isfile(os.path.join(bare, "HEAD")):
        # Existing cache: top up.  Re-apply safety config in case this
        # cache was created by an older version that didn't set it.
        _apply_safety_config(bare)
        _run_git(
            ["fetch", "--no-tags", "--no-auto-gc", "--update-shallow", real_src,
             "+refs/heads/*:refs/fetch-cache/heads/*",
             "+refs/remotes/origin/*:refs/fetch-cache/remotes/origin/*",
             "+refs/tags/*:refs/fetch-cache/tags/*"],
            cwd=bare,
        )
        _remove_shallow(bare)
        logger.info("updated %s.git <- %s", name, real_src)
        return bare

    # First time — init + fetch (never inherits shallow).
    r = _run_git(["init", "--bare", bare])
    if r.returncode != 0:
        logger.info("%s.git: init --bare failed, skipping", name)
        return None
    _apply_safety_config(bare)
    r = _run_git(
        ["fetch", "--update-shallow", real_src,
         "+refs/heads/*:refs/heads/*",
         "+refs/remotes/origin/*:refs/remotes/origin/*",
         "+refs/tags/*:refs/tags/*"],
        cwd=bare,
    )
    if r.returncode != 0:
        # Leave the partial bare in place; next update goes through the
        # "existing cache" path and resumes the fetch.
        logger.info("%s.git: fetch failed, will retry on next update", name)
        return None
    logger.info("created %s.git <- %s", name, real_src)
    return bare


# ---------------------------------------------------------------------------
# update — single-module mode (called by wrapper after each clone)
# ---------------------------------------------------------------------------

def cmd_update_src(cache_dir: str, src_dir: str) -> int:
    """Create or update a bare cache from a single *src_dir* and any of
    its submodule git dirs.

    Called by the ``FetchContent_MakeAvailable`` override the moment a
    populate finishes, so the cache fills up module-by-module as the
    build progresses.  If the build aborts midway, every module that
    has already been populated is still reusable on the next build.
    """
    if not os.path.isdir(src_dir):
        logger.info("src %s does not exist, nothing to update", src_dir)
        return 0
    os.makedirs(cache_dir, exist_ok=True)
    _ensure_cache(src_dir, cache_dir)
    _update_submodules(src_dir, cache_dir)
    return 0


# ---------------------------------------------------------------------------
# update — full-scan mode (manual rebuild / legacy fallback)
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
        _ensure_cache(src_dir, cache_dir)

        # Submodule repos (scan .git/modules/)
        _update_submodules(src_dir, cache_dir)

    logger.info("FetchContent cache update finished.")
    return 0


def _update_submodules(src_dir: str, cache_dir: str) -> None:
    """Cache bare repos found under ``.git/modules/`` (recursively)."""
    # Resolve the actual git dir (handles both .git/ dir and .git file)
    r = _run_git(["rev-parse", "--git-dir"], cwd=src_dir)
    if r.returncode != 0:
        return
    git_dir = os.path.join(src_dir, r.stdout.strip())
    modules_dir = os.path.join(git_dir, "modules")
    if os.path.isdir(modules_dir):
        _walk_submodule_dirs(modules_dir, cache_dir, rel_prefix="")


def _walk_submodule_dirs(modules_dir: str, cache_dir: str,
                         rel_prefix: str) -> None:
    """Cache every git repo reachable under ``modules_dir``.

    Children are either real submodule git dirs (HEAD + objects/) —
    cache them and recurse into their ``modules/`` — or path-segment
    directories when a submodule path contains a slash (e.g.
    ``third-party/fmt`` -> ``third-party/``) — recurse to find the git
    dirs inside.
    """
    try:
        it = os.scandir(modules_dir)
    except OSError:
        return
    with it:
        for entry in it:
            if not entry.is_dir(follow_symlinks=False):
                continue
            sub = entry.path
            rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            is_git_dir = (os.path.isfile(os.path.join(sub, "HEAD"))
                          and os.path.isdir(os.path.join(sub, "objects")))
            if is_git_dir:
                _ensure_cache(sub, cache_dir)
                nested = os.path.join(sub, "modules")
                if os.path.isdir(nested):
                    _walk_submodule_dirs(nested, cache_dir, rel)
            else:
                # Intermediate path segment (submodule path with a slash).
                _walk_submodule_dirs(sub, cache_dir, rel)


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
    mode = sp.add_mutually_exclusive_group(required=True)
    mode.add_argument("--src",
                      help="Single-module mode: path to a freshly-cloned "
                           "src repo (used by fetch_cache_wrapper.py)")
    mode.add_argument("--build-dir",
                      help="Full-scan mode: walk _deps/ under this cmake "
                           "build dir (manual rebuild / legacy fallback)")

    args = p.parse_args()

    if args.cmd == "update":
        if args.src:
            return cmd_update_src(args.cache_dir, args.src)
        return cmd_update(args.cache_dir, args.build_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
