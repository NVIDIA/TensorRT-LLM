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
"""Build and verify the BOLT profile bundle manifest.

The manifest pins a profile bundle to the exact bits it was generated from so
that phase 3 (apply) and offline customer re-BOLT can detect drift. It records:

  - arch / triple / glibc
  - LLVM/BOLT version used
  - source ref/commit
  - per-ELF hashes of the ORIGINAL (pre-bolt) libraries
  - workload list used for profiling
  - per-profile (.yaml/.fdata) checksums

Usage:
  manifest.py build  --work-dir DIR --profiles DIR [--ref REF] [--suite F] -o manifest.json
  manifest.py verify --work-dir DIR --profiles DIR manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

MANIFEST_VERSION = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_triple() -> str:
    machine = platform.machine()
    # Match the design doc's <triple> convention.
    return {"aarch64": "aarch64-linux-gnu", "x86_64": "x86_64-linux-gnu"}.get(
        machine, f"{machine}-linux-gnu"
    )


def detect_glibc() -> str:
    name, ver = platform.libc_ver()
    return f"{name}-{ver}" if ver else "unknown"


def detect_llvm_bolt_version() -> str:
    try:
        out = subprocess.run(
            ["llvm-bolt", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    for line in out.splitlines():
        if "LLVM version" in line:
            return line.split("LLVM version", 1)[1].strip()
    return out.strip().splitlines()[0] if out.strip() else "unknown"


def detect_ref(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_workloads(suite: Path | None) -> list[str]:
    if not suite or not suite.is_file():
        return []
    try:
        import yaml
    except ImportError:
        print("[WARNING] PyYAML unavailable; workload list omitted", file=sys.stderr)
        return []
    doc = yaml.safe_load(suite.read_text()) or {}
    return [wl.get("name", "") for wl in doc.get("workloads", []) or [] if wl.get("enabled", True)]


def select_workloads(workloads_arg: str | None, suite: Path | None) -> list[str]:
    """Workloads to record in the manifest.

    An explicit --workloads list (the workloads ACTUALLY profiled in this run)
    takes precedence over the suite declaration. The suite YAML lists every
    candidate workload -- including ones a given run did not exercise -- which
    would overstate the bundle's coverage; CI passes --workloads with exactly the
    fan-out set that produced .fdata.
    """
    if workloads_arg:
        return [w.strip() for w in workloads_arg.split(",") if w.strip()]
    return read_workloads(suite)


def hash_originals(work_dir: Path) -> dict[str, str]:
    original = work_dir / "original"
    if not original.is_dir():
        return {}
    return {
        p.name: sha256(p) for p in sorted(original.iterdir()) if p.is_file() and ".so" in p.name
    }


def hash_profiles(profiles_dir: Path) -> dict[str, str]:
    if not profiles_dir.is_dir():
        return {}
    out: dict[str, str] = {}
    for p in sorted(profiles_dir.iterdir()):
        if p.is_file() and p.suffix in (".yaml", ".fdata"):
            out[p.name] = sha256(p)
    return out


def cmd_build(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    profiles_dir = Path(args.profiles)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "arch": platform.machine(),
        "triple": detect_triple(),
        "glibc": detect_glibc(),
        "llvm_bolt_version": detect_llvm_bolt_version(),
        "ref": detect_ref(args.ref),
        "workloads": select_workloads(
            getattr(args, "workloads", None),
            Path(args.suite) if args.suite else None,
        ),
        "original_elf_sha256": hash_originals(work_dir),
        "profile_sha256": hash_profiles(profiles_dir),
    }
    out = Path(args.output)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Wrote manifest: {out}")
    print(
        f"[INFO]   arch={manifest['triple']} llvm={manifest['llvm_bolt_version']} "
        f"ref={manifest['ref']}"
    )
    print(
        f"[INFO]   {len(manifest['original_elf_sha256'])} ELF(s), "
        f"{len(manifest['profile_sha256'])} profile(s), "
        f"{len(manifest['workloads'])} workload(s)"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    work_dir = Path(args.work_dir)
    profiles_dir = Path(args.profiles)

    errors: list[str] = []

    cur_triple = detect_triple()
    if manifest.get("triple") != cur_triple:
        errors.append(
            f"arch mismatch: manifest={manifest.get('triple')} current={cur_triple} "
            "(profiles are NOT cross-arch reusable)"
        )

    cur_elf = hash_originals(work_dir)
    for name, want in manifest.get("original_elf_sha256", {}).items():
        got = cur_elf.get(name)
        if got is None:
            errors.append(f"missing ELF: {name}")
        elif got != want:
            errors.append(f"ELF hash mismatch: {name}")

    cur_prof = hash_profiles(profiles_dir)
    for name, want in manifest.get("profile_sha256", {}).items():
        got = cur_prof.get(name)
        if got is None:
            errors.append(f"missing profile: {name}")
        elif got != want:
            errors.append(f"profile checksum mismatch: {name}")

    if errors:
        print("[ERROR] Manifest verification FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("[INFO] Manifest verification OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a manifest.json")
    b.add_argument("--work-dir", required=True, help="$BOLT_WORK_DIR (has original/)")
    b.add_argument("--profiles", required=True, help="dir with .yaml/.fdata profiles")
    b.add_argument("--ref", default=None, help="source ref/commit (default: git HEAD)")
    b.add_argument("--suite", default=None, help="workload suite YAML (fallback for workload list)")
    b.add_argument(
        "--workloads",
        default=None,
        help="comma-separated names of workloads actually profiled "
        "(overrides --suite; use for CI to record the real fan-out set)",
    )
    b.add_argument("-o", "--output", default="manifest.json")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="verify ELFs/profiles against a manifest")
    v.add_argument("manifest")
    v.add_argument("--work-dir", required=True)
    v.add_argument("--profiles", required=True)
    v.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
