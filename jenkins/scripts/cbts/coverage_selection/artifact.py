# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve which post-merge CBTS touch DBs to use.

Candidates are recent builds of `<ARTIFACT_BASE>` with both x86 and SBSA
coverage tarballs. The newest complete pair wins. The PR diff must apply
cleanly to that DB's revision; otherwise Tier 2 declines rather than choosing
an older DB. Revision metadata comes from the forge compare API and needs
`GITHUB_API_TOKEN`, the anonymous quota being per-IP and exhausted by shared CI
egress.

The selected architecture DBs are merged locally through the compact coverage
schema, producing the one DB consumed by `main.py`.

Two entry points. `--print-selection` prints `{urls, build, commit, lag,
base_commit, drift, drift_status, patch_apply_status}` and stops. `--prepare
DIR` goes on to download, unpack, and merge the winner, drop that JSON beside
it, and print `{path, meta}` — the two paths `main.py` needs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coverage_utils"))

from compact_db import merge_databases  # noqa: E402

# Merged-artifact base for the main-branch L0_PostMerge job.
ARTIFACT_BASE = "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge"
ARCH_TARBALL_NAMES = (
    "cbts_pystart_report_x86_64.tar.gz",
    "cbts_pystart_report_SBSA.tar.gz",
)
# sqlite at the tar root, and the selection JSON `prepare` drops beside it.
DB_NAME = "cbts_touchmap.sqlite"
META_NAME = "cbts_coverage_db.json"
# Per-build metadata carrying `commit=<sha>`; absent on some builds.
BUILD_INFO_NAME = "build_info.txt"

# Branch the DB is collected from; must match ARTIFACT_BASE.
COVERAGE_BRANCH = "main"
# Read by `compare_distance`; the anonymous quota is unusable from shared CI egress IPs.
GITHUB_TOKEN_ENV = "GITHUB_API_TOKEN"

_URM = "https://urm.nvidia.com/artifactory"
_GITHUB_COMPARE = "https://api.github.com/repos/NVIDIA/TensorRT-LLM/compare"
_GITHUB_GIT_REPO = "https://github.com/NVIDIA/TensorRT-LLM.git"
_JENKINS_BASE = "https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge"
# Cover the 30-commit freshness window plus missing or unsuccessful builds.
_MAX_PROBE = 50
# Per-request timeout in seconds, for the small JSON/metadata calls.
_TIMEOUT = 15
# Socket timeout for the tarball itself, which runs to hundreds of MB.
_DOWNLOAD_TIMEOUT = 300
# Tarball download attempts.
_RETRIES = 3

_PatchApplyStatus = Literal["clean", "conflict", "unknown"]


def _get(url: str, headers: Optional[dict] = None) -> tuple[Optional[int], Optional[bytes]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, None
    except OSError as e:
        print(f"[artifact] error fetching {url}: {e}", file=sys.stderr)
        return None, None


def _exists(url: str) -> bool:
    """True if the artifact exists — a 1-byte ranged GET; 200/206 means present."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status in (200, 206)
    except urllib.error.HTTPError:
        return False
    except OSError as e:
        print(f"[artifact] error probing {url}: {e}", file=sys.stderr)
        return False


def latest_build_number(jenkins_base: str = _JENKINS_BASE) -> Optional[int]:
    """Newest build number via the Jenkins REST API (lastBuild, then lastCompletedBuild)."""
    for kind in ("lastBuild", "lastCompletedBuild"):
        status, data = _get(f"{jenkins_base}/{kind}/api/json")
        if status == 200 and data:
            try:
                return int(json.loads(data)["number"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return None


def tarball_url(build: int, name: str, artifact_base: str = ARTIFACT_BASE) -> str:
    return f"{_URM}/{artifact_base}/{build}/cbts-coverage/{name}"


def tarball_urls(build: int, artifact_base: str = ARTIFACT_BASE) -> list[str]:
    return [tarball_url(build, name, artifact_base) for name in ARCH_TARBALL_NAMES]


def build_info_url(build: int, artifact_base: str = ARTIFACT_BASE) -> str:
    return f"{_URM}/{artifact_base}/{build}/{BUILD_INFO_NAME}"


def build_commit(build: int, artifact_base: str = ARTIFACT_BASE) -> Optional[str]:
    """Revision a build ran, from its `build_info.txt`, or None when unavailable."""
    status, data = _get(build_info_url(build, artifact_base))
    if status != 200 or not data:
        return None
    for line in data.decode("utf-8", "replace").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "commit" and value.strip():
            return value.strip()
    return None


@lru_cache(maxsize=None)
def _compare(base: str, head: str) -> Optional[dict]:
    """The forge's `base...head` compare payload, or None when it cannot be had."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get(GITHUB_TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, data = _get(f"{_GITHUB_COMPARE}/{base}...{head}", headers)
    if status != 200 or not data:
        # 403 without a token means the shared egress IP burned the 60/h anonymous quota.
        hint = " (no token: anonymous quota)" if status == 403 and not token else ""
        print(
            f"[artifact] compare {base[:10]}...{head[:10]} failed: HTTP {status}{hint}",
            file=sys.stderr,
        )
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"[artifact] compare {base[:10]}...{head[:10]}: bad response: {e}", file=sys.stderr)
        return None


def compare_distance(commit: str, branch: str = COVERAGE_BRANCH) -> Optional[int]:
    """Commits `branch` gained since `commit` — reporting only; never negative."""
    payload = _compare(commit, branch)
    if payload is None:
        return None
    try:
        # `ahead_by` counts the full range; only the `commits` array is truncated at 250.
        return int(payload["ahead_by"])
    except (KeyError, TypeError, ValueError) as e:
        print(f"[artifact] compare {commit[:10]}...{branch}: no ahead_by: {e}", file=sys.stderr)
        return None


def merge_base(head: str, branch: str = COVERAGE_BRANCH) -> Optional[str]:
    """The commit `head` forked from `branch` — the revision the PR's diff is against."""
    payload = _compare(branch, head)
    if payload is None:
        return None
    sha = (payload.get("merge_base_commit") or {}).get("sha")
    if not sha:
        print(f"[artifact] compare {branch}...{head[:10]}: no merge_base_commit", file=sys.stderr)
        return None
    return sha


def drift(db_commit: str, base_commit: str) -> tuple[Optional[int], str]:
    """Distance from the DB's revision to the PR's base; direction is recorded, not weighted."""
    payload = _compare(db_commit, base_commit)
    if payload is None:
        return None, "unknown"
    try:
        distance = int(payload["ahead_by"]) + int(payload["behind_by"])
        return distance, str(payload.get("status") or "")
    except (KeyError, TypeError, ValueError) as e:
        print(
            f"[artifact] compare {db_commit[:10]}...{base_commit[:10]}: no ahead/behind: {e}",
            file=sys.stderr,
        )
        return None, "unknown"


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    input_data: Optional[bytes] = None,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )


def _patch_apply_status(
    pr_base_commit: str,
    pr_head: str,
    db_commit: str,
    repo_root: Path = Path("."),
    upstream_url: str = _GITHUB_GIT_REPO,
) -> _PatchApplyStatus:
    """Return clean/conflict/unknown for applying the PR diff to the DB revision."""
    repo_source = repo_root.resolve()
    checked_out_head = _run_git(["rev-parse", "HEAD"], cwd=repo_source)
    if (
        checked_out_head.returncode != 0
        or checked_out_head.stdout.decode("ascii", "replace").strip() != pr_head
    ):
        print(f"[artifact] checked-out revision is not PR head {pr_head[:10]}", file=sys.stderr)
        return "unknown"
    with tempfile.TemporaryDirectory(prefix="cbts_patch_check_") as temp_dir:
        temp = Path(temp_dir)
        git_dir = temp / "repo.git"
        initialized = _run_git(["init", "--bare", str(git_dir)], cwd=temp)
        if initialized.returncode != 0:
            print("[artifact] could not initialize the patch-check repository", file=sys.stderr)
            return "unknown"

        # Fetching into a temporary bare repository leaves the shallow CI checkout and its
        # index untouched. The checked-out PR head is available locally; both main revisions
        # are fetched from the public upstream repository.
        fetched_head = _run_git(
            [
                "--git-dir",
                str(git_dir),
                "fetch",
                "--no-tags",
                "--depth=1",
                str(repo_source),
                "HEAD",
            ],
            cwd=temp,
        )
        if fetched_head.returncode != 0:
            print(
                f"[artifact] could not load PR head {pr_head[:10]} for patch check",
                file=sys.stderr,
            )
            return "unknown"

        revisions = list(dict.fromkeys((pr_base_commit, db_commit)))
        fetched_revisions = _run_git(
            [
                "--git-dir",
                str(git_dir),
                "fetch",
                "--no-tags",
                "--depth=1",
                upstream_url,
                *revisions,
            ],
            cwd=temp,
        )
        if fetched_revisions.returncode != 0:
            print(
                "[artifact] could not load the base/coverage revisions for patch check",
                file=sys.stderr,
            )
            return "unknown"

        diff = _run_git(
            [
                "--git-dir",
                str(git_dir),
                "diff",
                "--binary",
                "--full-index",
                pr_base_commit,
                pr_head,
            ],
            cwd=temp,
        )
        if diff.returncode != 0:
            print("[artifact] could not generate the PR diff for patch check", file=sys.stderr)
            return "unknown"

        index = temp / "index"
        git_env = os.environ.copy()
        git_env["GIT_INDEX_FILE"] = str(index)
        read_tree = _run_git(
            ["--git-dir", str(git_dir), "read-tree", db_commit], cwd=temp, env=git_env
        )
        if read_tree.returncode != 0:
            print(
                "[artifact] could not read the coverage revision for patch check",
                file=sys.stderr,
            )
            return "unknown"

        applied = _run_git(
            [
                "--git-dir",
                str(git_dir),
                "apply",
                "--cached",
                "--3way",
                "--whitespace=nowarn",
                "-",
            ],
            cwd=temp,
            input_data=diff.stdout,
            env=git_env,
        )
        return "clean" if applied.returncode == 0 else "conflict"


def _selection_accepts_pr_diff(sel: dict, pr_base_commit: str, pr_head: str) -> bool:
    sel["patch_apply_status"] = _patch_apply_status(pr_base_commit, pr_head, sel["commit"])
    if sel["patch_apply_status"] == "clean":
        return True
    print(
        f"[artifact] coverage tier declined: PR diff does not apply cleanly to "
        f"latest DB commit {sel['commit'][:10]} ({sel['patch_apply_status']})",
        file=sys.stderr,
    )
    return False


def select_tarball(
    pr_base_commit: str,
    artifact_base: str = ARTIFACT_BASE,
    jenkins_base: str = _JENKINS_BASE,
    max_probe: int = _MAX_PROBE,
) -> Optional[dict]:
    """Newest complete architecture pair with measurable PR-base topology."""
    build = latest_build_number(jenkins_base)
    if build is None:
        print("[artifact] could not resolve latest build number", file=sys.stderr)
        return None
    for b in range(build, max(0, build - max_probe), -1):
        urls = tarball_urls(b, artifact_base)
        if not all(_exists(url) for url in urls):
            continue
        commit = build_commit(b, artifact_base)
        if not commit:
            print(f"[artifact] build {b}: commit unknown; skipped", file=sys.stderr)
            continue
        distance, status = drift(commit, pr_base_commit)
        if distance is None:
            print(
                f"[artifact] latest complete build {b}: topology against the PR base unknown",
                file=sys.stderr,
            )
            return None
        if status not in ("ahead", "behind", "identical"):
            print(
                f"[artifact] latest complete build {b}: relation to the PR base is "
                f"{status or 'unknown'}",
                file=sys.stderr,
            )
            return None
        selected = {
            "url": urls[0],
            "urls": urls,
            "build": b,
            "commit": commit,
            "base_commit": pr_base_commit,
            "drift": distance,
            "drift_status": status,
            "lag": compare_distance(commit),
        }
        if selected["lag"] is None:
            print(f"[artifact] build {b}: lag behind main unknown", file=sys.stderr)
        return selected
    print(
        f"[artifact] no complete x86/SBSA coverage pair in the last {max_probe} builds",
        file=sys.stderr,
    )
    return None


def measure_drift(sel: dict, pr_head: Optional[str]) -> dict:
    """Add PR-base relation metadata to a pinned selection, in place."""
    sel.setdefault("base_commit", None)
    sel.setdefault("drift", None)
    sel.setdefault("drift_status", "unknown")
    if not pr_head or not sel.get("commit"):
        return sel
    base = merge_base(pr_head)
    if not base:
        return sel
    sel["base_commit"] = base
    sel["drift"], sel["drift_status"] = drift(sel["commit"], base)
    return sel


def describe(sel: dict) -> str:
    """One-line account of the selection, for the CI log."""
    lag = "unknown" if sel.get("lag") is None else f"+{sel['lag']} commit(s)"
    if sel.get("drift") is None:
        base_distance = "unknown"
    else:
        base_distance = f"+{sel['drift']} commit(s)"
    base = (sel.get("base_commit") or "unknown")[:10]
    return (
        f"[artifact] selected build {sel.get('build')}, "
        f"DB commit {(sel.get('commit') or 'unknown')[:10]}; topology from DB: "
        f"PR base {base} ({base_distance}), current main tip ({lag})"
    )


def download(url: str, dest: Path, attempts: int = _RETRIES) -> Optional[Path]:
    """Stream the tarball into `dest`, retrying; None when every attempt fails.

    Streamed, not buffered: the artifact runs to hundreds of MB.
    """
    out = dest / url.rsplit("/", maxsplit=1)[-1]
    for attempt in range(1, attempts + 1):
        try:
            with (
                urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp,
                out.open("wb") as f,
            ):
                shutil.copyfileobj(resp, f)
            return out
        except OSError as e:  # HTTPError/URLError are OSError subclasses
            print(f"[artifact] download {attempt}/{attempts} failed: {e}", file=sys.stderr)
    return None


def extract(tarball: Path, dest: Path) -> bool:
    """Unpack the tarball into `dest`; False when it is not readable."""
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            # `filter` lands in 3.12 and is the default from 3.14; older runtimes lack it.
            if sys.version_info >= (3, 12):
                tf.extractall(dest, filter="data")
            else:
                tf.extractall(dest)
    except (OSError, tarfile.TarError) as e:
        print(f"[artifact] extract failed: {e}", file=sys.stderr)
        return False
    return True


def prepare(dest_dir: str, pr_head: Optional[str]) -> Optional[dict]:
    """Resolve, download, and merge the DB pair; `{path, meta}` or None on failure.

    `meta` is the selection JSON on disk, which `main.py --coverage-db-meta` reads.
    Paths are relative to the caller's cwd, matching the Groovy caller's `cd ${LLM_ROOT}`.
    """
    if not pr_head:
        print("[artifact] PR head is required to select a coverage DB", file=sys.stderr)
        return None
    pr_base_commit = merge_base(pr_head)
    if not pr_base_commit:
        print("[artifact] could not resolve the PR base commit", file=sys.stderr)
        return None
    sel = select_tarball(pr_base_commit)
    if sel is None:
        return None
    if not _selection_accepts_pr_diff(sel, pr_base_commit, pr_head):
        return None
    print(describe(sel), file=sys.stderr)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    db = dest / DB_NAME
    with tempfile.TemporaryDirectory(prefix="cbts_artifacts_", dir=dest) as temp_dir:
        temp = Path(temp_dir)
        source_dbs = []
        for index, url in enumerate(sel["urls"]):
            tarball = download(url, temp)
            artifact_dir = temp / str(index)
            artifact_dir.mkdir()
            if tarball is None or not extract(tarball, artifact_dir):
                return None
            source_db = artifact_dir / DB_NAME
            if not source_db.is_file():
                print(f"[artifact] {DB_NAME} not in {url}", file=sys.stderr)
                return None
            source_dbs.append(source_db)
        try:
            connection = merge_databases(source_dbs, db)
        except (OSError, sqlite3.Error, ValueError) as e:
            print(f"[artifact] merge failed: {e}", file=sys.stderr)
            return None
        connection.close()

    meta = dest / META_NAME
    meta.write_text(json.dumps(sel))
    return {"path": str(db), "meta": str(meta)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--print-selection",
        action="store_true",
        help="resolve and print {urls, build, commit, lag, base_commit, drift, "
        "drift_status, patch_apply_status} as JSON",
    )
    ap.add_argument(
        "--build", type=int, default=None, help="pin a build number (skip auto-resolve)"
    )
    ap.add_argument(
        "--prepare",
        metavar="DIR",
        default=None,
        help="resolve, download and merge the architecture DBs into DIR, then print "
        "{path, meta} as JSON",
    )
    ap.add_argument(
        "--pr-head",
        default=None,
        help="required PR head revision; its merge base measures DB drift and defines "
        "the diff checked against the latest coverage revision",
    )
    args = ap.parse_args(argv)

    if not args.print_selection and not args.prepare:
        ap.error("one of --print-selection / --prepare is required")

    if args.prepare:
        ready = prepare(args.prepare, args.pr_head)
        if ready is None:
            return 1
        print(json.dumps(ready))
        return 0

    if not args.pr_head:
        print("[artifact] --pr-head is required", file=sys.stderr)
        return 1
    pr_base_commit = merge_base(args.pr_head)
    if not pr_base_commit:
        return 1

    if args.build is not None:
        urls = tarball_urls(args.build)
        if not all(_exists(url) for url in urls):
            return 1
        commit = build_commit(args.build)
        best = {
            "url": urls[0],
            "urls": urls,
            "build": args.build,
            "commit": commit,
            "lag": compare_distance(commit) if commit else None,
        }
        measure_drift(best, args.pr_head)
        if best["drift"] is None or best["drift_status"] not in ("ahead", "behind", "identical"):
            return 1
    else:
        best = select_tarball(pr_base_commit)
    if best is None:
        return 1
    if not _selection_accepts_pr_diff(best, pr_base_commit, args.pr_head):
        return 1
    print(json.dumps(best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
