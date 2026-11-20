# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the perf reproduction registry and every entry's source.yaml.

Checks (offline): index.yaml lists every entry directory exactly once; each source.yaml has the required fields with
valid values; entry_id matches the directory; README.md and configs/ exist.

With --check-git, also checks that each entry's commit exists in the TensorRT-LLM repository and is reachable from a
branch that the repository ruleset protects against deletion and force-push (or from the entry's tag). Run it from a
TensorRT-LLM clone whose ``origin`` is https://github.com/NVIDIA/TensorRT-LLM.

Usage: python3 examples/perf_reproductions/_common/validate.py [--check-git] [--remote origin]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"branch", "partial", "mainline", "superseded"}
UPSTREAM_STATUSES = {"branch-only", "in-review", "equivalent-on-main", "mainline"}
# Branches excluded from the repository's all-branch-name-rules ruleset (deletion and force-push are allowed there).
UNPROTECTED_BRANCHES = {
    "feat/visual_gen",
    "feat/1.2-mlpinf",
    "feat/bench_x",
    "feat/rl_rc",
    "feat/k3_26.08",
}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _get(d: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def check_entry(entry_dir: Path) -> list[str]:
    errors = []
    rel = entry_dir.relative_to(ROOT).as_posix()
    source_path = entry_dir / "source.yaml"
    source = yaml.safe_load(source_path.read_text())
    for field in (
        "entry_id",
        "status",
        "benchmark.name",
        "model",
        "hardware",
        "code.repo",
        "code.commit",
        "image.name",
        "image.digest",
        "upstream",
    ):
        if _get(source, field) in (None, "", []):
            errors.append(f"{rel}: missing {field}")
    if source.get("entry_id") != rel:
        errors.append(f"{rel}: entry_id {source.get('entry_id')!r} does not match its directory")
    if source.get("status") not in STATUSES:
        errors.append(f"{rel}: status must be one of {sorted(STATUSES)}")
    commit = _get(source, "code.commit")
    if not isinstance(commit, str):
        # YAML reads an unquoted SHA made of digits (optionally with one "e") as a number.
        errors.append(f"{rel}: code.commit must be a quoted string, got {type(commit).__name__}")
    elif not SHA_RE.match(commit):
        errors.append(f"{rel}: code.commit must be a full 40-character SHA")
    if source.get("status") != "mainline" and not (
        _get(source, "code.branch") or _get(source, "code.tag")
    ):
        errors.append(f"{rel}: a non-mainline entry needs code.branch or code.tag")
    platforms = _get(source, "image.platform_digests") or {}
    for key in ["image.digest"] + [f"image.platform_digests.{p}" for p in platforms]:
        value = _get(source, key)
        if value is not None and not DIGEST_RE.match(str(value)):
            errors.append(f"{rel}: {key} is not a sha256 digest")
    for item in source.get("upstream") or []:
        if item.get("status") not in UPSTREAM_STATUSES:
            errors.append(
                f"{rel}: upstream {item.get('feature')!r} status must be one of {sorted(UPSTREAM_STATUSES)}"
            )
    for required in ("README.md", "configs"):
        if not (entry_dir / required).exists():
            errors.append(f"{rel}: missing {required}")
    return errors


def check_git(entry_dir: Path, remote: str) -> list[str]:
    source = yaml.safe_load((entry_dir / "source.yaml").read_text())
    rel = entry_dir.relative_to(ROOT).as_posix()
    commit, branch, tag = (
        _get(source, "code.commit"),
        _get(source, "code.branch"),
        _get(source, "code.tag"),
    )
    refs = []
    if tag:
        refs.append(f"refs/tags/{tag}")
    if branch:
        if branch in UNPROTECTED_BRANCHES and not tag:
            return [
                f"{rel}: branch {branch} is not protected against force-push or deletion; add code.tag"
            ]
        refs.append(f"refs/heads/{branch}")
    if source.get("status") == "mainline":
        refs.append("refs/heads/main")
    errors = []
    local = [f"refs/perf-repro-check/{i}" for i in range(len(refs))]
    subprocess.run(
        ["git", "fetch", "--quiet", remote] + [f"+{r}:{dst}" for r, dst in zip(refs, local)],
        check=False,
    )
    try:
        for ref, local_ref in zip(refs, local):
            ok = (
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", commit, local_ref], capture_output=True
                ).returncode
                == 0
            )
            if ok:
                return []
            errors.append(f"{rel}: commit {commit[:12]} is not reachable from {ref}")
    finally:
        for local_ref in local:
            subprocess.run(["git", "update-ref", "-d", local_ref], capture_output=True, check=False)
    return errors or [f"{rel}: no ref to check commit {commit[:12]} against"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check-git", action="store_true", help="also check commit reachability (needs network)"
    )
    parser.add_argument("--remote", default="origin", help="git remote of NVIDIA/TensorRT-LLM")
    args = parser.parse_args()
    index = yaml.safe_load((ROOT / "index.yaml").read_text())
    listed = [e["path"] for e in index["entries"]]
    on_disk = sorted(p.parent.relative_to(ROOT).as_posix() for p in ROOT.glob("*/**/source.yaml"))
    errors = []
    if sorted(listed) != on_disk or len(set(listed)) != len(listed):
        errors.append(
            f"index.yaml entries {sorted(listed)} do not match entry directories {on_disk}"
        )
    for rel in on_disk:
        entry_errors = check_entry(ROOT / rel)
        errors += entry_errors
        if args.check_git and not entry_errors:
            errors += check_git(ROOT / rel, args.remote)
    for e in errors:
        print(f"ERROR: {e}")
    print(f"{len(on_disk)} entries checked, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
