#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import sys


class PinnedMemoryUsageChecker(ast.NodeVisitor):
    def __init__(self, *, allow_direct_pin_memory: bool) -> None:
        self.allow_direct_pin_memory = allow_direct_pin_memory
        self.violations: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "pin_memory":
            if not self.allow_direct_pin_memory:
                self.violations.append(
                    (
                        node.lineno,
                        "Use `maybe_pin_memory(tensor)` instead of direct `.pin_memory()`.",
                    )
                )

        for keyword in node.keywords:
            if (
                keyword.arg == "pin_memory"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                self.violations.append(
                    (
                        node.lineno,
                        "Use `pin_memory=prefer_pinned()` instead of `pin_memory=True`.",
                    )
                )

        self.generic_visit(node)


def _check_file(path: pathlib.Path) -> list[tuple[int, str]]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [(0, f"Failed to read file: {exc}")]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"Failed to parse file: {exc.msg}")]

    allow_direct_pin_memory = path.as_posix().endswith("tensorrt_llm/_utils.py")
    checker = PinnedMemoryUsageChecker(allow_direct_pin_memory=allow_direct_pin_memory)
    checker.visit(tree)
    return checker.violations


def main(argv: list[str]) -> int:
    if len(argv) <= 1:
        return 0

    has_violations = False
    for file_arg in argv[1:]:
        path = pathlib.Path(file_arg)
        violations = _check_file(path)
        for lineno, message in violations:
            has_violations = True
            print(f"{path}:{lineno}: {message}")

    if has_violations:
        print("\nPinned-memory policy check failed.")
        print("Use `tensorrt_llm._utils.maybe_pin_memory()` for direct pinning.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
