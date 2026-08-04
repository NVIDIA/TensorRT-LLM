#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enforce import discipline inside ``tensorrt_llm/_torch/auto_deploy``.

Two rules:

A. Imports that resolve inside the auto_deploy package MUST be relative.
   Absolute forms like ``import tensorrt_llm._torch.auto_deploy.X`` or
   ``from tensorrt_llm._torch.auto_deploy.X import Y`` are forbidden.

B. Relative imports MUST stay inside the auto_deploy package. A relative
   import that escapes (e.g. ``from ....llmapi import X``) is forbidden;
   reach the rest of TensorRT-LLM via absolute ``tensorrt_llm.X`` imports
   instead.

These rules keep auto_deploy's source tree portable: it can be copied
verbatim into the standalone ``llmc`` package without rewriting any
in-package import statements.
"""

import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
AD_ROOT = REPO_ROOT / "tensorrt_llm" / "_torch" / "auto_deploy"
AD_PKG = "tensorrt_llm._torch.auto_deploy"


def _file_package_parts(path: pathlib.Path) -> list[str]:
    """Return the dotted package path of the directory containing *path*."""
    rel = path.resolve().relative_to(REPO_ROOT)
    parts = list(rel.parts[:-1])  # drop filename
    return parts


def _check_file(path: pathlib.Path) -> list[tuple[int, str]]:
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        # Treat parse failures as violations rather than silently passing —
        # a malformed file should never sneak through the discipline check.
        return [(exc.lineno or 1, f"failed to parse file: {exc.msg}")]

    pkg_parts = _file_package_parts(path)
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == AD_PKG or alias.name.startswith(AD_PKG + "."):
                    violations.append(
                        (
                            node.lineno,
                            f"absolute self-import `import {alias.name}` — "
                            f"use a relative import instead",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                # Absolute import.
                mod = node.module or ""
                if mod == AD_PKG or mod.startswith(AD_PKG + "."):
                    violations.append(
                        (
                            node.lineno,
                            f"absolute self-import `from {mod} import ...` — "
                            f"use a relative import instead",
                        )
                    )
            else:
                # Relative import — make sure it stays inside auto_deploy.
                if node.level > len(pkg_parts):
                    violations.append(
                        (
                            node.lineno,
                            f"relative import with level={node.level} escapes the package root",
                        )
                    )
                    continue
                # Resolve the target package the relative import points to.
                # `from .foo import bar` at depth 0 -> pkg_parts (the file's pkg)
                # `from ..foo import bar` -> pkg_parts[:-1]
                base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                base = ".".join(base_parts)
                if not (base == AD_PKG or base.startswith(AD_PKG + ".")):
                    target = base + (("." + node.module) if node.module else "")
                    violations.append(
                        (
                            node.lineno,
                            f"relative import resolves to `{target}` which is "
                            f"outside `{AD_PKG}` — use an absolute "
                            f"`from tensorrt_llm.<...> import ...` instead",
                        )
                    )

    return violations


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        targets = [pathlib.Path(p).resolve() for p in argv[1:]]
    else:
        targets = sorted(AD_ROOT.rglob("*.py"))

    ad_root_resolved = AD_ROOT.resolve()
    failures: list[tuple[pathlib.Path, int, str]] = []
    for path in targets:
        try:
            path.resolve().relative_to(ad_root_resolved)
        except ValueError:
            # File passed in by pre-commit that isn't under auto_deploy — skip.
            continue
        for lineno, msg in _check_file(path):
            failures.append((path, lineno, msg))

    if failures:
        print(
            "Import discipline check failed for tensorrt_llm/_torch/auto_deploy/.",
            file=sys.stderr,
        )
        for path, lineno, msg in failures:
            try:
                rel = path.resolve().relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            print(f"  {rel}:{lineno}: {msg}", file=sys.stderr)
        print(
            "\nFix: replace absolute `tensorrt_llm._torch.auto_deploy.X` imports "
            "with relative imports (e.g. `from ..X import Y`). For paths outside "
            "auto_deploy, use absolute `tensorrt_llm.<...>` imports instead of "
            "relative imports that escape the package.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
