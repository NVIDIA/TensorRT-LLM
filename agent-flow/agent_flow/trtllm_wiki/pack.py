# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, cache and load a wiki pack: the facts of one checkout state, as files.

A pack is a directory::

    manifest.json   schema version, repository, commit, dirty-tree digest,
                    build time, per-fact counts and the facts that came back empty
    facts/*.json    one file per fact kind (see FACTS)
    index.md        the overview an agent reads first (render.py)

A pack belongs to one checkout state: the commit plus a digest of the
uncommitted changes. :func:`ensure` reuses a pack whose state and schema
match and rebuilds otherwise, so a run that resumes on a moved checkout gets
the facts of the tree it now runs on.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from . import accuracy, config, env, modeling_v2, recipes, tests

#: Bump when a fact's shape changes: older packs are rebuilt, never misread.
WIKI_SCHEMA = 1

FACTS: dict[str, Callable[[Path], Any]] = {
    "config": config.extract_config_fields,
    "families": modeling_v2.extract_families,
    "ops": modeling_v2.extract_ops,
    "test_lists": tests.extract_test_lists,
    "waives": tests.extract_waives,
    "ci_stages": tests.extract_stages,
    "accuracy_tasks": accuracy.extract_accuracy_tasks,
    "accuracy_references": accuracy.extract_references,
    "recipes": recipes.extract_recipes,
    "env_vars": env.extract_env_vars,
}


def _git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "GIT_PAGER": "cat", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


#: The directories the facts are read from; changes elsewhere (a run directory
#: inside the checkout, this pack) do not make a pack stale.
SOURCE_ROOTS = ("tensorrt_llm", "tests", "examples", "cpp", "jenkins")


def source_state(repo: Path) -> dict[str, str]:
    """``{"commit", "dirty"}``: the checkout's HEAD and a digest of uncommitted source changes."""
    commit = _git(repo, "rev-parse", "HEAD").strip()
    roots = ("--", *SOURCE_ROOTS)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all", *roots)
    dirty = ""
    if status.strip():
        digest = hashlib.sha1(status.encode())
        digest.update(_git(repo, "diff", "HEAD", *roots).encode())
        # An untracked file (a new family, list or target) changes the facts too.
        others = _git(repo, "ls-files", "--others", "--exclude-standard", "-z", *roots)
        for name in others.split("\0"):
            try:
                stat = (Path(repo) / name).stat() if name else None
            except OSError:
                continue
            if stat is not None:
                digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
        dirty = digest.hexdigest()[:12]
    return {"commit": commit, "dirty": dirty}


def _count(value: Any) -> int:
    if isinstance(value, dict) and "fields" in value:
        return len(value["fields"])
    return len(value) if isinstance(value, (list, dict)) else 0


def build(repo: Path, out: Path) -> dict[str, Any]:
    """Extract every fact of ``repo`` into ``out`` and return the manifest."""
    from .render import render_index

    repo, out = Path(repo).resolve(), Path(out)
    started = time.monotonic()
    (out / "facts").mkdir(parents=True, exist_ok=True)
    facts: dict[str, Any] = {}
    for name, extract in FACTS.items():
        facts[name] = extract(repo)
        target = out / "facts" / f"{name}.json"
        target.write_text(json.dumps(facts[name], indent=1, sort_keys=True), encoding="utf-8")
    counts = {name: _count(value) for name, value in facts.items()}
    manifest = {
        "schema": WIKI_SCHEMA,
        "repo": str(repo),
        **source_state(repo),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_s": round(time.monotonic() - started, 2),
        "counts": counts,
        "empty": sorted(name for name, count in counts.items() if not count),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    from .query import Wiki

    (out / "index.md").write_text(render_index(Wiki(out)), encoding="utf-8")
    return manifest


def read_manifest(out: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def current(repo: Path, out: Path) -> bool:
    """Whether the pack in ``out`` holds the facts of ``repo`` as it is now."""
    manifest = read_manifest(out)
    if not manifest or manifest.get("schema") != WIKI_SCHEMA:
        return False
    state = source_state(Path(repo))
    return bool(state["commit"]) and all(manifest.get(k) == v for k, v in state.items())


def ensure(repo: Path, out: Path, *, force: bool = False) -> tuple[dict[str, Any], bool]:
    """``(manifest, rebuilt)``: the pack in ``out`` for ``repo``, built when it is not current."""
    if not force and current(repo, out):
        manifest = read_manifest(out)
        assert manifest is not None
        return manifest, False
    return build(repo, out), True


def default_cache_dir(repo: Path) -> Path:
    """Where a pack for ``repo`` is kept when no directory is named: per commit and tree state."""
    state = source_state(Path(repo))
    key = f"{state['commit'][:12] or 'nocommit'}{'-' + state['dirty'] if state['dirty'] else ''}"
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "trtllm-wiki" / key
