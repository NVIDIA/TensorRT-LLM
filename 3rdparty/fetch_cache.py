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

Called automatically by cmake (init) and build_wheel.py (update).

Subcommands
-----------
  init    Create bare reference repos for each git dependency
  update  Merge build-time _deps objects back into the cache
"""

import argparse
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_TIMEOUT = 600  # seconds

# Minimal safety config applied to every bare reference repo.
# Prevents accidental GC of objects that clones reference via alternates.
SAFETY_CONFIG = {
    "gc.auto": "0",
    "gc.pruneExpire": "never",
}


# ---------------------------------------------------------------------------
# Dependency discovery
# ---------------------------------------------------------------------------

def read_dependencies(json_path: str) -> list[dict]:
    """
    Read git dependencies from fetch_content.json.

    Handles both the current schema (``{"dependencies": [...]}`` with
    ``schema_version``) and older flat-list formats.

    Returns ``[{"name": ..., "url": ...}, ...]``, excluding URL-based
    (tarball) dependencies.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read %s: %s", json_path, exc)
        return []

    if isinstance(data, dict) and "dependencies" in data:
        raw = data["dependencies"]
    elif isinstance(data, list):
        raw = data
    else:
        logger.warning("Unrecognised fetch_content.json schema")
        return []

    deps = []
    for dep in raw:
        if not isinstance(dep, dict):
            continue
        name = dep.get("name", "")
        url = dep.get("git_repository", "") or dep.get("git_url", "")
        if dep.get("use_url"):
            continue
        if not name or not url:
            continue
        url = re.sub(r"\$\{?\{?github_base_url\}?\}?",
                      "https://github.com", url)
        deps.append({"name": name, "url": url})
    return deps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_name(url: str) -> str:
    """Extract repo name from URL: ``https://x/y/Foo.git`` -> ``Foo``."""
    return os.path.basename(url.rstrip("/")).removesuffix(".git")


def _remove_partial(path: str) -> None:
    """Remove a partially-cloned bare repo (best-effort)."""
    import shutil
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except OSError:
        pass


def _run_git(args, **kwargs):
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    # Never prompt for credentials — use existing credential helpers / SSH
    # keys, but if the repo doesn't exist or auth fails, fail immediately.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if "env" in kwargs:
        env.update(kwargs["env"])
    kwargs["env"] = env
    return subprocess.run(["git"] + args, **kwargs)


# ---------------------------------------------------------------------------
# init — create bare reference repos
# ---------------------------------------------------------------------------

def _init_one(url: str, ref_dir: str, lock_dir: str) -> str | None:
    """
    Init a single bare reference repo with file locking.

    Returns the ref_dir on success, None on failure.
    """
    name = _repo_name(url)

    if os.path.isfile(os.path.join(ref_dir, "HEAD")):
        return ref_dir  # already done

    lock_path = os.path.join(lock_dir, f".{name}.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        deadline = time.monotonic() + LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    logger.warning("  %s: lock timeout", name)
                    return None
                time.sleep(2)

        # Double-check after acquiring lock
        if os.path.isfile(os.path.join(ref_dir, "HEAD")):
            return ref_dir

        logger.info("  %s: cloning from %s", name, url)
        try:
            result = _run_git(["clone", "--bare", url, ref_dir])
        except subprocess.TimeoutExpired:
            logger.info("  %s: clone timed out, skipping", name)
            _remove_partial(ref_dir)
            return None
        if result.returncode != 0:
            # Repo may not exist or auth failed — silently skip for
            # compatibility (some URLs in fetch_content.json may be
            # invalid or private).
            logger.info("  %s: clone failed, skipping", name)
            _remove_partial(ref_dir)
            return None

        # Apply safety config
        for key, val in SAFETY_CONFIG.items():
            _run_git(["config", key, val], cwd=ref_dir)

        return ref_dir

    except Exception:
        _remove_partial(ref_dir)
        return None
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def cmd_init(json_path: str, cache_dir: str, jobs: int = 4) -> int:
    """Init subcommand: create bare cache repos for all deps."""
    deps = read_dependencies(json_path)
    if not deps:
        logger.warning("No git dependencies found in %s", json_path)
        return 0  # not an error

    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Initializing FetchContent cache (%d deps) -> %s",
                len(deps), cache_dir)

    tasks = []
    for dep in deps:
        name = _repo_name(dep["url"])
        ref_dir = os.path.join(cache_dir, f"{name}.git")
        if os.path.isfile(os.path.join(ref_dir, "HEAD")):
            logger.info("  %s: cached", name)
            continue
        tasks.append((dep["url"], ref_dir))

    if not tasks:
        logger.info("All dependencies already cached")
        return 0

    failed = 0
    with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
        futures = {
            pool.submit(_init_one, url, ref_dir, cache_dir): _repo_name(url)
            for url, ref_dir in tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            result = future.result()
            if result:
                logger.info("  %s: done", name)
            else:
                failed += 1

    return 0  # always succeed — missing cache entries are silently skipped


# ---------------------------------------------------------------------------
# update — merge build _deps back into cache
# ---------------------------------------------------------------------------

def cmd_update(
    cache_dir: str,
    build_dir: str,
    json_path: str | None = None,
) -> int:
    """Update subcommand: merge build artifacts into cache."""
    deps_dir = os.path.join(build_dir, "_deps")
    if not os.path.isdir(deps_dir):
        logger.info("No _deps directory in %s, nothing to update", build_dir)
        return 0

    if not os.path.isdir(cache_dir):
        logger.warning("Cache dir does not exist: %s", cache_dir)
        return 0

    # Build name -> URL mapping for cache lookup
    name_to_url: dict[str, str] = {}
    if json_path and os.path.isfile(json_path):
        for dep in read_dependencies(json_path):
            name_to_url[dep["name"]] = dep["url"]

    merged = 0
    for entry in sorted(os.listdir(deps_dir)):
        if not entry.endswith("-src"):
            continue
        src_dir = os.path.join(deps_dir, entry)
        git_marker = os.path.join(src_dir, ".git")
        if not (os.path.isdir(git_marker) or os.path.isfile(git_marker)):
            continue

        cmake_name = entry.removesuffix("-src")
        ref_dir = _find_cache_entry(cache_dir, cmake_name, name_to_url, src_dir)
        if not ref_dir:
            continue

        logger.info("  Merging %s -> %s", cmake_name, os.path.basename(ref_dir))
        result = _run_git(
            ["-c", "transfer.fsckObjects=true",
             "fetch", "--no-tags", "--no-auto-gc",
             os.path.realpath(src_dir),
             "+refs/heads/*:refs/fetch-cache/heads/*",
             "+refs/tags/*:refs/fetch-cache/tags/*"],
            cwd=ref_dir,
        )
        if result.returncode == 0:
            merged += 1
        else:
            logger.warning("  %s: merge failed: %s",
                           cmake_name, result.stderr.strip())

    logger.info("Merged %d repos into cache", merged)
    return 0


def _find_cache_entry(
    cache_dir: str,
    cmake_name: str,
    name_to_url: dict[str, str],
    src_dir: str,
) -> str | None:
    """Resolve a cmake dep name to its cache directory."""
    # 1) Via fetch_content.json name -> URL -> basename
    if cmake_name in name_to_url:
        name = _repo_name(name_to_url[cmake_name])
        ref = os.path.join(cache_dir, f"{name}.git")
        if os.path.isdir(ref):
            return ref

    # 2) Via git remote in the fetched repo
    try:
        r = _run_git(["config", "--get", "remote.origin.url"],
                      cwd=src_dir, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            name = _repo_name(r.stdout.strip())
            ref = os.path.join(cache_dir, f"{name}.git")
            if os.path.isdir(ref):
                return ref
    except (subprocess.TimeoutExpired, OSError):
        pass

    # 3) Direct name match
    ref = os.path.join(cache_dir, f"{cmake_name}.git")
    if os.path.isdir(ref):
        return ref

    return None


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

    sp = sub.add_parser("init", help="Create bare reference cache repos")
    sp.add_argument("json_path", help="Path to fetch_content.json")
    sp.add_argument("--cache-dir", required=True, help="Cache directory")
    sp.add_argument("--jobs", type=int, default=4,
                    help="Parallel clone jobs (default: 4)")

    sp = sub.add_parser("update", help="Merge build artifacts into cache")
    sp.add_argument("--cache-dir", required=True, help="Cache directory")
    sp.add_argument("--build-dir", required=True, help="CMake build directory")
    sp.add_argument("--json", dest="json_path",
                    help="Path to fetch_content.json (for name mapping)")

    args = p.parse_args()

    if args.cmd == "init":
        return cmd_init(args.json_path, args.cache_dir, args.jobs)
    elif args.cmd == "update":
        return cmd_update(args.cache_dir, args.build_dir, args.json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
