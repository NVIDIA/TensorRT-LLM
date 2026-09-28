# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate, patch and stage the FA4 Python/JIT sources shipped in TRT-LLM."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

VERSION = "4.0.0b19"
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
    destination.mkdir(parents=True)
    for path in sorted((source / "flash_attn/cute").glob("*.py")):
        content = path.read_text().replace("flash_attn.cute", "trtllm_flash_attn")
        if path.name == "__init__.py":
            start = content.index("from importlib.metadata")
            end = content.index("from .interface")
            content = (
                content[:start]
                + "from ._build_info import VERSION as __version__\n\n"
                + content[end:]
            )
        (destination / path.name).write_text(content)
    for name in ("LICENSE", "AUTHORS"):
        shutil.copy2(source / name, destination / name)
    patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    (destination / "_build_info.py").write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        f"VERSION = {VERSION!r}\n"
        f"SOURCE_REVISION = {VALIDATED_REVISION!r}\n"
        f"PATCH_SHA256 = {patch_digest!r}\n"
        f"BUILD_ID = {(VERSION, VALIDATED_REVISION, PATCHED_SHA256, patch_digest)!r}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--apply-patch", action="store_true")
    args = parser.parse_args()
    prepare(
        args.source,
        Path(__file__).resolve().parent,
        apply_patch=args.apply_patch,
        destination=args.destination,
    )
