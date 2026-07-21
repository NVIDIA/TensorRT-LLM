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

The producer's post-merge `Test Coverage` stage merges every stage's PY_START
files into `cbts_touchmap.sqlite`, tars it as `cbts_pystart_report.tar.gz`
(sqlite at the tar root plus `cbts_report/`), and uploads it to
`<ARTIFACT_BASE>/<build>/cbts-coverage/` (see L0_MergeRequest.groovy Test
Coverage stage; ARTIFACT_BASE mirrors L0_Test.groovy UPLOAD_PATH).

`latest_tarball_url()` finds the newest post-merge build that actually has the
tarball: it reads the latest build number from the Jenkins REST API (the only
in-repo precedent, get_image_key_to_tag.py) then walks builds down, probing
Artifactory with a 1-byte ranged GET until one exists — a build can be SUCCESS
yet upload nothing (no PY_START files -> early return), so `lastSuccessfulBuild`
alone is wrong.

Two entry points for the Groovy wiring:
  * `--print-url` — resolve and print the tarball URL only (no 1 GB download),
    so the caller can `wget` it through the CI's proven download path.
  * `--dest DIR` — download + extract, printing the local sqlite path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Base for the merged artifact — mirrors L0_Test.groovy UPLOAD_PATH
# (`sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}`) for the
# main-branch L0_PostMerge job. Reads resolve through the virtual repo, same as
# the cbts_test_db download in L0_Test.groovy.
ARTIFACT_BASE = "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge"
TARBALL_NAME = "cbts_pystart_report.tar.gz"
SQLITE_NAME = "cbts_touchmap.sqlite"

_URM = "https://urm.nvidia.com/artifactory"
_JENKINS_BASE = "https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge"
# How far back to walk when recent builds have no tarball (keeps a bounded probe).
_MAX_PROBE = 50
# Per-request timeout so one stalled endpoint can't hang the whole probe walk.
_TIMEOUT = 15


def _get(url: str) -> tuple[Optional[int], Optional[bytes]]:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001 — network best-effort; caller falls back
        print(f"[artifact] error fetching {url}: {e}", file=sys.stderr)
        return None, None


def _exists(url: str) -> bool:
    """True if the artifact exists — a 1-byte ranged GET (works where HEAD may not).

    HEAD is not guaranteed on Artifactory; the proven precedent uses GET. `Range:
    bytes=0-0` keeps the body ~1 byte (not the full 1 GB); a present artifact
    answers 206 (or 200 if the range is ignored).
    """
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status in (200, 206)
    except urllib.error.HTTPError as e:
        return e.code in (200, 206)
    except Exception as e:  # noqa: BLE001
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


def latest_tarball_url(
    artifact_base: str = ARTIFACT_BASE,
    jenkins_base: str = _JENKINS_BASE,
    max_probe: int = _MAX_PROBE,
) -> Optional[str]:
    """URL of the newest build whose coverage tarball actually exists, or None."""
    build = latest_build_number(jenkins_base)
    if build is None:
        print("[artifact] could not resolve latest build number", file=sys.stderr)
        return None
    floor = max(0, build - max_probe)
    while build > floor:
        url = tarball_url(build, artifact_base)
        if _exists(url):
            return url
        print(f"[artifact] build {build} has no tarball, trying {build - 1}", file=sys.stderr)
        build -= 1
    print(f"[artifact] no tarball in the last {max_probe} builds", file=sys.stderr)
    return None


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

    `url` pins an explicit tarball (skips latest-build resolution). Best-effort:
    any failure returns None so the caller falls back to coverage-off.
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
    except Exception as e:  # noqa: BLE001
        print(f"[artifact] download failed {url}: {e}", file=sys.stderr)
        return None
    return extract_touch_db(tarball, dest_dir)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dest", help="download + extract into DIR; prints the local sqlite path")
    ap.add_argument(
        "--print-url", action="store_true", help="resolve and print the tarball URL only"
    )
    ap.add_argument(
        "--build", type=int, default=None, help="pin a build number (skip auto-resolve)"
    )
    args = ap.parse_args(argv)

    url = tarball_url(args.build) if args.build is not None else None

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
