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
"""Phase-3 BOLT apply / repack (the "BOLTed build" step).

Takes a phase-1 BOLT-compatible release tarball (the original, relocs-carrying
ELFs) plus a phase-2 profile bundle (per-ELF .yaml/.fdata + manifest), runs
`llvm-bolt` on each in-scope ELF, substitutes the bolted binaries back into the
wheel AND the TensorRT-LLM/ layout, and repacks a new tarball under a bolted
TARNAME. No recompile.

Scope is defined by the bundle: an ELF is bolted iff a matching profile
(`<lib>.yaml`, falling back to `<lib>.fdata`) exists. The wheel's RECORD is
regenerated for any modified members so `pip install` stays consistent.

  apply_bolt.py --tarball TensorRT-LLM-GH200.tar.gz \
                --profiles /path/to/_merged \
                --output bolt-TensorRT-LLM-GH200.tar.gz \
                [--manifest manifest.json] [--strip] [--dry-run]

llvm-bolt must be on PATH (same version used to instrument/merge). The optimize
flags mirror scripts/bolt/bolt_lib.sh::optimize_libraries -- keep them in sync.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Keep in sync with bolt_lib.sh::optimize_libraries.
DEFAULT_BOLT_FLAGS = [
    "-lite",
    "-infer-stale-profile",
    "-reorder-blocks=ext-tsp",
    "-reorder-functions=hfsort",
    "-split-functions",
    "-split-all-cold",
    "-split-eh",
    "-dyno-stats",
]


def log(msg: str) -> None:
    print(f"[apply_bolt] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[apply_bolt][ERROR] {msg}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# Profile matching
# ---------------------------------------------------------------------------
def profile_for(elf_name: str, profiles_dir: Path) -> Path | None:
    """Return the profile (.yaml preferred, else .fdata) for an ELF basename.

    Profiles are named by the ELF basename with a single trailing '.so' removed
    (matching how bolt_lib.sh names merged profiles), e.g.
      libtensorrt_llm.so                        -> libtensorrt_llm.yaml
      bindings.cpython-312-aarch64-linux-gnu.so -> bindings.cpython-312-aarch64-linux-gnu.yaml
    """
    key = elf_name[:-3] if elf_name.endswith(".so") else elf_name
    for ext in (".yaml", ".fdata"):
        cand = profiles_dir / f"{key}{ext}"
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    return None


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# BOLT one ELF (in place)
# ---------------------------------------------------------------------------
def bolt_elf(elf: Path, profile: Path, flags: list[str], strip: bool, dry_run: bool) -> bool:
    """BOLT `elf` in place using `profile`. Returns True if optimized."""
    rel = elf.name
    if dry_run:
        log(f"  would bolt {rel}  <-  {profile.name}")
        return True

    out = elf.with_suffix(elf.suffix + ".bolted")
    cmd = ["llvm-bolt", str(elf), "-o", str(out), f"-data={profile}"] + flags
    log(f"  bolting {rel}  <-  {profile.name}")
    rc, output = run(cmd)
    if rc != 0:
        err(f"llvm-bolt failed for {rel} (rc={rc})")
        for line in output.splitlines()[-15:]:
            err(f"    {line}")
        out.unlink(missing_ok=True)
        return False

    if strip:
        rc_s, _ = run(["llvm-strip", "--strip-all", str(out)])
        if rc_s != 0:
            err(f"llvm-strip failed for {rel}; keeping unstripped bolted binary")

    # Preserve mode, then replace original.
    out.chmod(elf.stat().st_mode)
    os.replace(out, elf)
    return True


# ---------------------------------------------------------------------------
# Wheel handling (unzip -> bolt matching members -> fix RECORD -> rezip)
# ---------------------------------------------------------------------------
def _record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={b64}"


def repack_wheel(
    root: Path, dest: Path, original_infos: dict[str, zipfile.ZipInfo]
) -> None:
    """Zip <root> into <dest>, re-stamping each member's original ZipInfo.

    zf.write() would record the on-disk mode, which extractall() already reduced
    to the umask default -- silently dropping the exec bit off any executable the
    wheel ships. So carry the source archive's external_attr (mode) and timestamp
    across for every member, bolted or not; only the content differs. Members
    absent from `original_infos` were created here and have no mode to preserve.
    """
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            src_info = original_infos.get(rel)
            if src_info is None:
                zf.write(f, rel)
                continue
            info = zipfile.ZipInfo(rel, date_time=src_info.date_time)
            info.external_attr = src_info.external_attr
            info.internal_attr = src_info.internal_attr
            info.create_system = src_info.create_system
            info.compress_type = zipfile.ZIP_DEFLATED
            # Stream rather than read the member into memory: the bolted
            # members include multi-hundred-MB shared objects.
            with zf.open(info, "w", force_zip64=True) as dst, f.open("rb") as src:
                shutil.copyfileobj(src, dst)


def process_wheel(
    wheel: Path, profiles_dir: Path, flags: list[str], strip: bool, dry_run: bool
) -> int:
    """Bolt matching libs inside a wheel and regenerate its RECORD.

    Returns the number of bolted members.
    """
    log(f"Wheel: {wheel.name}")
    with tempfile.TemporaryDirectory(prefix="bolt_whl_") as td:
        wd = Path(td)
        with zipfile.ZipFile(wheel) as zf:
            # extractall() applies the process umask rather than the archived
            # mode, so the source zip's central directory is the ONLY record of
            # each member's real permission bits. Keep it to re-stamp on repack.
            original_infos = {info.filename: info for info in zf.infolist()}
            zf.extractall(wd)

        bolted = 0
        changed: list[Path] = []
        for f in sorted(wd.rglob("*")):
            if not f.is_file() or not is_elf(f):
                continue
            prof = profile_for(f.name, profiles_dir)
            if prof is None:
                continue
            if bolt_elf(f, prof, flags, strip, dry_run):
                bolted += 1
                changed.append(f)

        if bolted == 0:
            log("  no matching ELFs in wheel; leaving it unchanged")
            return 0
        if dry_run:
            return bolted

        # Regenerate RECORD for the changed members.
        record_path = None
        for f in wd.rglob("RECORD"):
            if f.parent.name.endswith(".dist-info"):
                record_path = f
                break
        if record_path is None:
            err("wheel has no dist-info/RECORD; cannot safely repack")
            raise SystemExit(2)
        _rewrite_record(record_path, wd, changed)

        tmp_whl = wheel.with_suffix(".whl.new")
        repack_wheel(wd, tmp_whl, original_infos)
        os.replace(tmp_whl, wheel)
        log(f"  repacked wheel ({bolted} libs bolted, RECORD updated)")
        return bolted


def _rewrite_record(record_path: Path, wheel_root: Path, changed: list[Path]) -> None:
    changed_rel = {p.relative_to(wheel_root).as_posix() for p in changed}
    rows_out: list[list[str]] = []
    with record_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rel = row[0]
            if rel in changed_rel:
                fp = wheel_root / rel
                rows_out.append([rel, _record_hash(fp), str(fp.stat().st_size)])
            else:
                rows_out.append(row)
    with record_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows_out)


# ---------------------------------------------------------------------------
# Manifest verification (optional)
# ---------------------------------------------------------------------------
def verify_manifest(manifest_path: Path, tree: Path, strict: bool) -> None:
    manifest = json.loads(manifest_path.read_text())
    want = manifest.get("original_elf_sha256", {})
    if not want:
        log("manifest has no original ELF hashes; skipping verify")
        return
    # Hash one copy of each named ELF found in the tree (first match).
    found: dict[str, str] = {}
    for f in tree.rglob("*"):
        if f.is_file() and f.name in want and f.name not in found and is_elf(f):
            found[f.name] = sha256_file(f)
    mismatches = []
    for name, h in want.items():
        got = found.get(name)
        if got is None:
            mismatches.append(f"missing in tarball: {name}")
        elif got != h:
            mismatches.append(f"hash mismatch (rebuilt since profiling?): {name}")
    if mismatches:
        msg = "manifest verification problems:\n  - " + "\n  - ".join(mismatches)
        if strict:
            err(msg)
            raise SystemExit(2)
        log("WARNING: " + msg)
        log("(continuing; -infer-stale-profile will absorb drift. Use --strict to fail.)")
    else:
        log(f"manifest verify OK ({len(want)} ELF hashes matched)")


# ---------------------------------------------------------------------------
# Bundle / tarball helpers
# ---------------------------------------------------------------------------
def resolve_profiles(profiles: Path, workdir: Path) -> Path:
    """If `profiles` is a .tar.zst/.tar.gz bundle, extract it; return a dir."""
    if profiles.is_dir():
        return profiles
    if profiles.is_file() and profiles.name.endswith((".tar.zst", ".tar.gz", ".tgz")):
        dest = workdir / "profiles"
        dest.mkdir(parents=True, exist_ok=True)
        log(f"Extracting profile bundle {profiles.name}")
        flag = "--zstd" if profiles.name.endswith(".tar.zst") else "-z"
        rc, out = run(["tar", flag, "-xf", str(profiles), "-C", str(dest)])
        if rc != 0:
            err(f"failed to extract bundle: {out}")
            raise SystemExit(2)
        # Bundle may contain a single subdir; flatten if so.
        entries = list(dest.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
        return dest
    err(f"--profiles must be a directory or a .tar.zst/.tar.gz bundle: {profiles}")
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tarball",
        required=True,
        type=Path,
        help="phase-1 BOLT-compatible release tarball (TensorRT-LLM*.tar.gz)",
    )
    ap.add_argument(
        "--profiles",
        required=True,
        type=Path,
        help="profile dir (with <lib>.yaml) or a .tar.zst/.tar.gz bundle",
    )
    ap.add_argument("--output", required=True, type=Path, help="path for the new bolted tarball")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional manifest.json to verify ELF hashes against",
    )
    ap.add_argument("--strict", action="store_true", help="fail (not warn) on manifest mismatch")
    ap.add_argument(
        "--strip", action="store_true", help="llvm-strip each bolted ELF after optimization"
    )
    ap.add_argument(
        "--workdir", type=Path, default=None, help="scratch dir (default: a temp dir, cleaned up)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list which ELFs would be bolted; no llvm-bolt, no repack",
    )
    args = ap.parse_args()

    if not args.dry_run and shutil.which("llvm-bolt") is None:
        err("llvm-bolt not on PATH")
        return 2
    if not args.tarball.is_file():
        err(f"tarball not found: {args.tarball}")
        return 2

    owns_workdir = args.workdir is None
    workdir = Path(tempfile.mkdtemp(prefix="apply_bolt_")) if owns_workdir else args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        profiles_dir = resolve_profiles(args.profiles, workdir)
        log(
            f"Profiles: {profiles_dir} "
            f"({len(list(profiles_dir.glob('*.yaml')))} yaml, "
            f"{len(list(profiles_dir.glob('*.fdata')))} fdata)"
        )

        # Extract the tarball.
        extract = workdir / "extract"
        extract.mkdir(parents=True, exist_ok=True)
        log(f"Extracting {args.tarball.name}")
        rc, out = run(["tar", "-xf", str(args.tarball), "-C", str(extract)])
        if rc != 0:
            err(f"failed to extract tarball: {out}")
            return 2
        roots = [p for p in extract.iterdir() if p.is_dir()]
        tree = roots[0] if len(roots) == 1 else extract
        log(f"Tarball root: {tree.name}")

        if args.manifest:
            verify_manifest(args.manifest, tree, args.strict)

        total = 0
        # 1) Loose ELFs in the layout (benchmarks/cpp, triton_backend, etc.).
        for f in sorted(tree.rglob("*")):
            if f.suffix == ".whl" or not f.is_file() or not is_elf(f):
                continue
            prof = profile_for(f.name, profiles_dir)
            if prof is None:
                continue
            if bolt_elf(f, prof, DEFAULT_BOLT_FLAGS, args.strip, args.dry_run):
                total += 1

        # 2) Wheel(s).
        for wheel in sorted(tree.rglob("tensorrt_llm-*.whl")):
            total += process_wheel(
                wheel, profiles_dir, DEFAULT_BOLT_FLAGS, args.strip, args.dry_run
            )

        if total == 0:
            err("no ELFs matched a profile -- nothing bolted. Check --profiles names.")
            return 2
        log(f"Bolted {total} ELF(s) total (loose + wheel).")

        if args.dry_run:
            log("dry-run: skipping repack.")
            return 0

        # Repack the tarball under the new name.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log(f"Repacking -> {args.output}")
        rc, out = run(["tar", "-C", str(extract), "-czf", str(args.output), tree.name])
        if rc != 0:
            err(f"repack failed: {out}")
            return 2
        log(f"Done. Bolted tarball: {args.output}")
        return 0
    finally:
        if owns_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
