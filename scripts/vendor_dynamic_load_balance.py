# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the scheduler package and MegaMoE inference closure, replaying recorded patches."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n\n"
)


def revision(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _source_bytes(source: Path, *, header: bool) -> bytes:
    content = source.read_bytes().rstrip(b"\n") + b"\n"
    if header and b"Copyright" not in content[:2000]:
        content = HEADER.encode() + content
    return content


def copy_source(source: Path, target: Path, *, header: bool) -> None:
    content = _source_bytes(source, header=header)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != content:
        target.write_bytes(content)


def _read_downstream_patches(target: Path, source: Path) -> list[dict]:
    manifest = json.loads((target / "VENDOR_MANIFEST.json").read_text())
    patches = manifest.get("downstream_patches", [])
    for patch in patches:
        patch_path = target / patch["patch"]
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != patch["patch_sha256"]:
            raise RuntimeError(f"Downstream patch digest mismatch: {patch_path}")
        for file in patch.get("files", [patch]):
            content = _source_bytes(source / file["file"], header=True)
            if hashlib.sha256(content).hexdigest() != file["unpatched_vendored_sha256"]:
                raise RuntimeError(
                    f"Upstream source changed for {file['file']}; reconcile the downstream patch "
                    "and manifest before re-vendoring."
                )
    return patches


def _apply_downstream_patches(root: Path, target: Path, patches: list[dict]) -> None:
    for patch in patches:
        patch_path = target / patch["patch"]
        formatter = patch.get("format_before")
        if formatter:
            version = subprocess.check_output(["ruff", "--version"], text=True).strip()
            if version != formatter:
                raise RuntimeError(f"Downstream patch requires {formatter}; found {version}")
            subprocess.run(
                ["ruff", "format", "--config", str(root / "pyproject.toml"), str(target)],
                cwd=root,
                check=True,
            )
        subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=root, check=True)
        subprocess.run(["git", "apply", str(patch_path)], cwd=root, check=True)
        for file in patch.get("files", [patch]):
            digest = hashlib.sha256((target / file["file"]).read_bytes()).hexdigest()
            if digest != file["patched_sha256"]:
                raise RuntimeError(f"Patched source digest mismatch: {file['file']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", required=True, type=Path)
    parser.add_argument("--megamoe", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target_root = root / "tensorrt_llm/_torch/cute_dsl_kernels"
    scheduler_src = args.scheduler / "megamoe_scheduler"
    scheduler_dst = target_root / "megamoe_scheduler_v2"
    mega_src = args.megamoe / "next/sources"
    mega_dst = target_root / "cutedsl_megamoe"
    patches = _read_downstream_patches(mega_dst, mega_src)
    scheduler_files = []
    for source in sorted(scheduler_src.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        rel = source.relative_to(scheduler_src)
        copy_source(source, scheduler_dst / rel, header=False)
        scheduler_files.append(str(rel))

    pending = [
        path.relative_to(mega_dst) for path in mega_dst.rglob("*.py") if path.name != "__init__.py"
    ]
    pending.extend(
        (
            Path("kernel_src/schedulers/__init__.py"),
            Path("kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_kernel.py"),
        )
    )
    copied = set()
    while pending:
        rel = pending.pop()
        if rel in copied:
            continue
        source = mega_src / rel
        if not source.is_file():
            raise RuntimeError(f"Vendored inference module disappeared upstream: {rel}")
        copy_source(source, mega_dst / rel, header=True)
        copied.add(rel)
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = rel.parent
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                base = base.joinpath(*node.module.split("."))
            candidates = [base.with_suffix(".py"), base / "__init__.py"]
            candidates.extend((base / name.name).with_suffix(".py") for name in node.names)
            for candidate in candidates:
                if not (mega_src / candidate).is_file():
                    continue
                if candidate.name == "__init__.py" and (mega_dst / candidate).exists():
                    continue
                pending.append(candidate)

    _apply_downstream_patches(root, mega_dst, patches)
    for repo, target, files, kind, upstream in (
        (
            args.scheduler,
            scheduler_dst,
            scheduler_files,
            "complete scheduler package",
            "https://gitlab-master.nvidia.com/jintaop/cutedsl_eplb_scheduler_copy.git",
        ),
        (
            args.megamoe,
            mega_dst,
            [str(path) for path in sorted(copied)],
            "inference closure",
            "https://gitlab-master.nvidia.com/jintaop/cutedsl_megamoe.git",
        ),
    ):
        source_subdir = "next/sources" if repo == args.megamoe else "megamoe_scheduler"
        source_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain", "--", source_subdir], text=True
            ).strip()
        )
        record = {
            "commit": revision(repo),
            "source_tree_dirty": source_dirty,
            "kind": kind,
            "files": {
                name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in files
            },
        }
        if target == mega_dst and patches:
            record.update(
                source_tree_dirty_scope=(
                    "Upstream checkout used for vendoring only; downstream changes are recorded separately."
                ),
                downstream_modified=True,
                downstream_patches=patches,
            )
        (target / "VENDOR_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")
        downstream = "downstream_modified: true\n" if target == mega_dst and patches else ""
        (target / "VENDOR_STAMP").write_text(
            f"upstream: {upstream}\ncommit: {record['commit']}\n"
            f"source_tree_dirty: {str(source_dirty).lower()}\nkind: {kind}\n"
            f"{downstream}manifest: VENDOR_MANIFEST.json\n"
            "Regenerate with scripts/vendor_dynamic_load_balance.py; recorded downstream patches are replayed.\n"
        )
        print(f"{target.name}: {record['commit']} ({len(files)} files)")


if __name__ == "__main__":
    main()
