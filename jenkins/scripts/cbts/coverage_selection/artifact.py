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
"""Resolve and fetch the latest merged CBTS touch DB from Artifactory.

The tarball is uploaded per post-merge run to
`<ARTIFACT_BASE>/<build>/cbts-coverage/cbts_pystart_report.tar.gz` (sqlite at
the tar root plus `cbts_report/`).

`select_tarball()` reads the newest build number from the Jenkins REST API, walks
builds down probing Artifactory with a 1-byte ranged GET, and ranks the ones that
exist by how far `COVERAGE_BRANCH` has moved past the revision each collected
(`build_info.txt`), since a build can be a re-run of an older commit. That
distance comes from local git, or from the forge compare API when the CI
checkout is too shallow to answer; the build number is only a tie-break.

Three entry points for the Groovy wiring:
  * `--print-url` — resolve and print the tarball URL only (no download).
  * `--print-selection` — print `{url, build, commit, lag}` as JSON.
  * `--dest DIR` — download + extract, printing the local sqlite path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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
SQLITE_NAME = "cbts_touchmap.sqlite"
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
# Per-request timeout in seconds.
_TIMEOUT = 15


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


def commit_distance(commit: str, repo_root: str = ".", ref: str = "HEAD") -> Optional[int]:
    """Commits in `ref` not reachable from `commit`, or None if git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "rev-list", "--count", f"{commit}..{ref}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None


@lru_cache(maxsize=None)
def compare_distance(commit: str, branch: str = COVERAGE_BRANCH) -> Optional[int]:
    """Commits `branch` gained since `commit`, from the forge compare API, or None."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get(GITHUB_TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, data = _get(f"{_GITHUB_COMPARE}/{commit}...{branch}", headers)
    if status != 200 or not data:
        # 403 without a token means the shared egress IP burned the 60/h anonymous quota.
        hint = " (no token: anonymous quota)" if status == 403 and not token else ""
        print(
            f"[artifact] compare {commit[:10]}...{branch} failed: HTTP {status}{hint}",
            file=sys.stderr,
        )
        return None
    try:
        # `ahead_by` counts the full range; only the `commits` array is truncated at 250.
        return int(json.loads(data)["ahead_by"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"[artifact] compare {commit[:10]}...{branch}: bad response: {e}", file=sys.stderr)
        return None


def db_lag(commit: str, repo_root: str = ".") -> Optional[int]:
    """How far `COVERAGE_BRANCH` moved past `commit`: local git first, then the compare API."""
    lag = commit_distance(commit, repo_root)
    return lag if lag is not None else compare_distance(commit)


def select_tarball(
    artifact_base: str = ARTIFACT_BASE,
    jenkins_base: str = _JENKINS_BASE,
    max_probe: int = _MAX_PROBE,
    repo_root: str = ".",
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
        lag = db_lag(commit, repo_root) if commit else None
        candidates.append({"url": url, "build": b, "commit": commit, "lag": lag})
    if not candidates:
        print(f"[artifact] no tarball in the last {max_probe} builds", file=sys.stderr)
        return None
    # Known lag first (smallest = closest to HEAD); unknown lag falls back to build order.
    best = min(candidates, key=lambda c: (c["lag"] is None, c["lag"] or 0, -c["build"]))
    if best["lag"] is None:
        print(
            f"[artifact] build {best['build']}: commit unknown, selected by build number",
            file=sys.stderr,
        )
    skipped = [c["build"] for c in candidates if c["build"] > best["build"]]
    if skipped:
        print(
            f"[artifact] builds {skipped} carry an older commit than {best['build']}; skipped",
            file=sys.stderr,
        )
    return best


def latest_tarball_url(
    artifact_base: str = ARTIFACT_BASE,
    jenkins_base: str = _JENKINS_BASE,
    max_probe: int = _MAX_PROBE,
) -> Optional[str]:
    """URL of the best available coverage tarball; see `select_tarball`."""
    best = select_tarball(artifact_base, jenkins_base, max_probe)
    return best["url"] if best else None


def extract_touch_db(tarball: Path | str, dest_dir: Path | str) -> Optional[Path]:
    """Extract `cbts_touchmap.sqlite` from a downloaded tarball; return its path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith(SQLITE_NAME)), None)
        if member is None:
            return None
        member.name = SQLITE_NAME
        tf.extract(member, dest_dir)
    return dest_dir / SQLITE_NAME


def fetch_latest_touch_db(dest_dir: Path | str, url: Optional[str] = None) -> Optional[Path]:
    """Download + extract the latest post-merge touch DB; return local sqlite Path or None.

    `url` pins an explicit tarball (skips latest-build resolution); any failure returns None.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = url or latest_tarball_url()
    if url is None:
        return None
    tarball = dest_dir / TARBALL_NAME
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp, open(tarball, "wb") as f:
            shutil.copyfileobj(resp, f)
        return extract_touch_db(tarball, dest_dir)
    except OSError as e:
        print(f"[artifact] download/extract failed {url}: {e}", file=sys.stderr)
        return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dest", help="download + extract into DIR; prints the local sqlite path")
    ap.add_argument(
        "--print-url", action="store_true", help="resolve and print the tarball URL only"
    )
    ap.add_argument(
        "--print-selection",
        action="store_true",
        help="resolve and print {url, build, commit, lag} as JSON",
    )
    ap.add_argument(
        "--build", type=int, default=None, help="pin a build number (skip auto-resolve)"
    )
    ap.add_argument("--repo-root", default=".", help="repo the lag is measured against")
    args = ap.parse_args(argv)

    url = tarball_url(args.build) if args.build is not None else None

    if args.print_selection:
        if args.build is not None:
            commit = build_commit(args.build)
            best = {
                "url": url,
                "build": args.build,
                "commit": commit,
                "lag": db_lag(commit, args.repo_root) if commit else None,
            }
        else:
            best = select_tarball(repo_root=args.repo_root)
        if best is None:
            return 1
        print(json.dumps(best))
        return 0

    if args.print_url:
        url = url or latest_tarball_url()
        if url is None:
            return 1
        print(url)
        return 0

    if args.dest:
        path = fetch_latest_touch_db(args.dest, url=url)
        if path is None:
            return 1
        print(path)
        return 0

    ap.error("one of --print-url or --dest is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
