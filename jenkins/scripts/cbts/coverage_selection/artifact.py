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
"""Resolve which post-merge CBTS touch DB to use.

Candidates are the recent builds of `<ARTIFACT_BASE>` that have a
`cbts-coverage/cbts_pystart_report.tar.gz`, ranked by how far `COVERAGE_BRANCH`
has moved past the revision each collected (`build_info.txt`), since a build can
be a re-run of an older commit; the build number is only a tie-break. That
distance is `ahead_by` from the forge compare API — the CI checkout is depth-1,
so git cannot answer it — and needs `GITHUB_API_TOKEN`, the anonymous quota
being per-IP and exhausted by shared CI egress.

Ranking scores candidates against the tip of `COVERAGE_BRANCH`; gating scores the
winner against the PR's merge base, which `--pr-head` supplies.

Two entry points. `--print-selection` prints `{url, build, commit, lag,
base_commit, drift, drift_status}` and stops. `--prepare DIR` goes on to
download and unpack the winner, drop that JSON beside it, and print
`{path, meta}` — the two paths `main.py` needs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Merged-artifact base for the main-branch L0_PostMerge job.
ARTIFACT_BASE = "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge"
TARBALL_NAME = "cbts_pystart_report.tar.gz"
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
_JENKINS_BASE = "https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge"
# Max builds to walk back when recent builds have no tarball.
_MAX_PROBE = 10
# Per-request timeout in seconds, for the small JSON/metadata calls.
_TIMEOUT = 15
# Socket timeout for the tarball itself, which runs to hundreds of MB.
_DOWNLOAD_TIMEOUT = 300
# Tarball download attempts.
_RETRIES = 3


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


def tarball_url(build: int, artifact_base: str = ARTIFACT_BASE) -> str:
    return f"{_URM}/{artifact_base}/{build}/cbts-coverage/{TARBALL_NAME}"


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
    """Commits `branch` gained since `commit` — ranking only; never negative."""
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


def select_tarball(
    artifact_base: str = ARTIFACT_BASE,
    jenkins_base: str = _JENKINS_BASE,
    max_probe: int = _MAX_PROBE,
) -> Optional[dict]:
    """Tarball of the least-trailing commit as {url, build, commit, lag}; build number breaks ties."""
    build = latest_build_number(jenkins_base)
    if build is None:
        print("[artifact] could not resolve latest build number", file=sys.stderr)
        return None
    candidates = []
    for b in range(build, max(0, build - max_probe), -1):
        url = tarball_url(b, artifact_base)
        if not _exists(url):
            continue
        commit = build_commit(b, artifact_base)
        lag = compare_distance(commit) if commit else None
        candidates.append({"url": url, "build": b, "commit": commit, "lag": lag})
    if not candidates:
        print(f"[artifact] no tarball in the last {max_probe} builds", file=sys.stderr)
        return None
    # Known lag first (smallest = closest to the branch tip); unknown falls back to build order.
    best = min(candidates, key=lambda c: (c["lag"] is None, c["lag"] or 0, -c["build"]))
    if best["lag"] is None:
        print(
            f"[artifact] build {best['build']}: lag unknown, selected by build number",
            file=sys.stderr,
        )
    skipped = [c["build"] for c in candidates if c["build"] > best["build"]]
    if skipped:
        print(
            f"[artifact] builds {skipped} carry an older commit than {best['build']}; skipped",
            file=sys.stderr,
        )
    return best


def measure_drift(sel: dict, pr_head: Optional[str]) -> dict:
    """Add `base_commit` / `drift` / `drift_status` to the ranked winner, in place."""
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
    lag = "lag unknown" if sel.get("lag") is None else f"{sel['lag']} commit(s) behind main"
    if sel.get("drift") is None:
        drifted = "drift unmeasured"
    else:
        base = (sel.get("base_commit") or "")[:10]
        drifted = f"{sel['drift']} commit(s) {sel.get('drift_status')} the PR base {base}"
    return (
        f"[artifact] build {sel.get('build')}, commit {(sel.get('commit') or 'unknown')[:10]}, "
        f"{lag}, {drifted}"
    )


def download(url: str, dest: Path, attempts: int = _RETRIES) -> Optional[Path]:
    """Stream the tarball into `dest`, retrying; None when every attempt fails.

    Streamed, not buffered: the artifact runs to hundreds of MB.
    """
    out = dest / TARBALL_NAME
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
    """Resolve, measure, download and unpack the DB; `{path, meta}` or None on any failure.

    `meta` is the selection JSON on disk, which `main.py --coverage-db-meta` reads.
    Paths are relative to the caller's cwd, matching the Groovy caller's `cd ${LLM_ROOT}`.
    """
    sel = select_tarball()
    if sel is None:
        return None
    measure_drift(sel, pr_head)
    print(describe(sel), file=sys.stderr)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tarball = download(sel["url"], dest)
    if tarball is None or not extract(tarball, dest):
        return None
    db = dest / DB_NAME
    if not db.is_file():
        print(f"[artifact] {DB_NAME} not in the tarball", file=sys.stderr)
        return None

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
        help="resolve and print {url, build, commit, lag, base_commit, drift, drift_status} as JSON",
    )
    ap.add_argument(
        "--build", type=int, default=None, help="pin a build number (skip auto-resolve)"
    )
    ap.add_argument(
        "--prepare",
        metavar="DIR",
        default=None,
        help="resolve, download and unpack the DB into DIR, then print {path, meta} as JSON",
    )
    ap.add_argument(
        "--pr-head",
        default=None,
        help="PR head revision; its merge base is what drift is measured against "
        "(omitted leaves drift null, which the selector declines on).",
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

    if args.build is not None:
        commit = build_commit(args.build)
        best = {
            "url": tarball_url(args.build),
            "build": args.build,
            "commit": commit,
            "lag": compare_distance(commit) if commit else None,
        }
    else:
        best = select_tarball()
    if best is None:
        return 1
    print(json.dumps(measure_drift(best, args.pr_head)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
