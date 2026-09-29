# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate FA4 sources and build a patched wheel with upstream import paths."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

VERSION = "4.0.0b19"
WHEEL_VERSION = "4.0.0b19+trtllm.1"
VALIDATED_REVISION = "940cd9680f3315f2f06b43ab5bea2c2cf2d96806"
SOURCE_SHA256 = "a91cde0ab8839d2492684fc4b83a8018534a055632789eb8cc532db2fee29d8a"
PATCHED_SHA256 = "a156456035a11159ee7ae4f5d75528c774e76560507971f50885e184c8ad2dd4"

PATCH_SHA256 = "be63535ef5ea95bc3f3f1c9c68e06e47ed781c47d6f283fec20d094f3a7bf492"


def source_digest(source: Path) -> str:
    """Hash the exact upstream package inputs, including dependency metadata."""
    digest = hashlib.sha256()
    paths = [
        *source.glob("flash_attn/cute/*.py"),
        source / "flash_attn/cute/pyproject.toml",
        source / "LICENSE",
        source / "AUTHORS",
    ]
    for path in sorted(paths):
        content = path.read_bytes()
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def prepare(
    source: Path, thirdparty: Path, *, apply_patch: bool = False, destination: Path | None = None
) -> None:
    """Reject stale pins/patches before modifying sources or emitting a package."""
    dependencies = json.loads((thirdparty / "fetch_content.json").read_text())["dependencies"]
    dependency = next(dep for dep in dependencies if dep["name"] == "flash_attn_4")
    if dependency["git_tag"] != VALIDATED_REVISION:
        raise RuntimeError(
            "FA4 revision changed. Revalidate the patch and update prepare_fa4.py before building."
        )
    patch = thirdparty / dependency["patch_file"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != PATCH_SHA256:
        raise RuntimeError(
            "FA4 patch changed. Revalidate it and update prepare_fa4.py before building."
        )
    digest = source_digest(source)
    if apply_patch and digest == SOURCE_SHA256:
        subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "--fuzz=0", "-i", str(patch.resolve())],
            cwd=source,
            check=True,
        )
        digest = source_digest(source)
    if digest != PATCHED_SHA256:
        raise RuntimeError(
            "FA4 source/patch mismatch. Reconfigure from clean FA4 sources or revalidate the "
            "dependency and update prepare_fa4.py; refusing to package unvalidated FA4."
        )
    # Also validate the currently selected patch on incremental CMake builds.
    subprocess.run(
        [
            "patch",
            "-p1",
            "--reverse",
            "--force",
            "--dry-run",
            "--fuzz=0",
            "-i",
            str(patch.resolve()),
        ],
        cwd=source,
        check=True,
    )
    if destination is None:
        return
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
    package = destination / "flash_attn" / "cute"
    package.mkdir(parents=True)
    for path in sorted((source / "flash_attn/cute").glob("*.py")):
        shutil.copy2(path, package / path.name)
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(source / "flash_attn/cute" / name, package / name)
    for name in ("LICENSE", "AUTHORS"):
        shutil.copy2(source / name, package / name)
    (package / "_trtllm_build_info.py").write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        f"VERSION = {WHEEL_VERSION!r}\n"
        f"SOURCE_REVISION = {VALIDATED_REVISION!r}\n"
        f"PATCH_SHA256 = {PATCH_SHA256!r}\n"
        f"BUILD_ID = {(WHEEL_VERSION, VALIDATED_REVISION, PATCHED_SHA256, PATCH_SHA256)!r}\n"
    )


def build_wheel(project: Path, thirdparty: Path, wheel_dir: Path) -> Path:
    """Build with upstream metadata and the exact patched runtime version."""
    runtime_pin = f"flash-attn-4=={WHEEL_VERSION}"
    if runtime_pin not in (thirdparty.parent / "requirements-fa4.txt").read_text().splitlines():
        raise RuntimeError("FA4 runtime pin changed; update the validated wheel version together.")
    bootstrap_pin = f"flash-attn-4=={VERSION}"
    if bootstrap_pin not in (thirdparty.parent / "requirements.txt").read_text().splitlines():
        raise RuntimeError("FA4 bootstrap pin changed; revalidate the source revision and patch.")
    wheel_dir.mkdir(parents=True, exist_ok=True)
    for old_wheel in wheel_dir.glob("flash_attn_4-*.whl"):
        old_wheel.unlink()
    env = os.environ.copy()
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4"] = WHEEL_VERSION
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir.resolve()),
            str(project / "flash_attn/cute"),
        ],
        env=env,
        check=True,
    )
    wheel = wheel_dir / f"flash_attn_4-{WHEEL_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = f"flash_attn_4-{WHEEL_VERSION}.dist-info/METADATA"
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        if metadata["Name"] != "flash-attn-4" or metadata["Version"] != WHEEL_VERSION:
            raise RuntimeError("FA4 wheel metadata does not match the patched dependency pin.")
        for path in (project / "flash_attn/cute").glob("*.py"):
            if archive.read(f"flash_attn/cute/{path.name}") != path.read_bytes():
                raise RuntimeError(f"FA4 wheel contains stale source: {path.name}")
        for name in ("LICENSE", "AUTHORS"):
            if not any(p.endswith(f"/{name}") for p in archive.namelist()):
                raise RuntimeError(f"FA4 wheel is missing {name}")
    return wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--apply-patch", action="store_true")
    parser.add_argument("--wheel-dir", type=Path)
    args = parser.parse_args()
    if args.wheel_dir is not None and args.destination is None:
        parser.error("--wheel-dir requires --destination")
    prepare(
        args.source,
        Path(__file__).resolve().parent,
        apply_patch=args.apply_patch,
        destination=args.destination,
    )
    if args.wheel_dir is not None:
        build_wheel(args.destination, Path(__file__).resolve().parent, args.wheel_dir)
