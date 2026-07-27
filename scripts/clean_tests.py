# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import ast
import os
import re
import subprocess
from collections import defaultdict

DEFAULT_TEST_DIR = "tests"
DEFAULT_IGNORE_DIRS = {
    "perf",
    "ray*",
    "cpp",
    "thirdparty",
    "stress_test",
    "sysinfo",
    "triton_server",
}
DEFAULT_IGNORE_FILES = {"test_unittests.py", "test_list_validation.py", "test_list_parser.py"}

CONFTEST = "conftest.py"

PYTEST_HOOKS = {
    "pytest_addoption",
    "pytest_configure",
    "pytest_collection_modifyitems",
    "pytest_generate_tests",
    "pytest_runtest_makereport",
    "pytest_runtest_protocol",
    "pytest_sessionstart",
    "pytest_sessionfinish",
    "pytest_unconfigure",
}


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[run_cmd] command failed (exit {result.returncode}): {cmd}", flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
    return result.stdout


def _dir_ignored(name, ignore_dirs):
    """Check if a directory name matches any entry in ignore_dirs.

    Supports exact names ('perf') and prefix wildcards ('ray*').
    """
    for pattern in ignore_dirs:
        if pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True
    return False


def collect_py_files(test_dir, ignore_dirs, ignore_files=None):
    """Walk test_dir and return list of .py file paths, skipping ignore_dirs and ignore_files."""
    ignore_files = ignore_files or set()
    py_files = []
    for root, dirs, files in os.walk(test_dir):
        dirs[:] = [d for d in dirs if not _dir_ignored(d, ignore_dirs)]
        for fname in files:
            if fname.endswith(".py") and fname not in ignore_files:
                py_files.append(os.path.join(root, fname))
    return py_files


# =========================
# Step 1: Get valid tests from test_lists directory
# =========================
def _parse_test_name(line):
    """Return canonical test ID from a test list line, without parametrize brackets.

    Handles both .txt format ('path.py::Class::test[param]')
    and .yml list items ('- path.py::test[param] TIMEOUT(120)').
    Returns 'path.py::Class::test' or 'path.py::test', or None if no test identifier.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("- "):
        line = line[2:].strip()
    token = line.split()[0]
    if "::" not in token:
        return None
    parts = token.split("::")
    if "[" in parts[-1]:
        parts[-1] = parts[-1][: parts[-1].index("[")]
    return "::".join(parts)


def get_used_tests(test_lists_dir):
    print(f"Collecting tests from test lists in: {test_lists_dir}")
    used = set()

    for root, _, files in os.walk(test_lists_dir):
        for fname in files:
            if not (fname.endswith(".txt") or fname.endswith(".yml")):
                continue
            filepath = os.path.join(root, fname)
            with open(filepath, "r") as f:
                for line in f:
                    name = _parse_test_name(line)
                    if name:
                        used.add(name)

    return used


# =========================
# Step 2: Remove test functions that are not collected
# =========================
def make_unused_test_predicate(used_full_ids, filepath, test_dir):
    relpath = os.path.relpath(filepath, test_dir)
    prefix = relpath + "::"
    suffix_template = "::{name}"

    def should_remove(node):
        if not node.name.startswith("test_"):
            return False
        # Match exact top-level: relpath::func
        if f"{relpath}::{node.name}" in used_full_ids:
            return False
        # Match class-qualified: relpath::*::func
        for uid in used_full_ids:
            if uid.startswith(prefix) and uid.endswith(suffix_template.format(name=node.name)):
                return False
        print(f"[REMOVE TEST] {node.name}")
        return True

    return should_remove


# =========================
# Step 3: Use vulture to find unused functions
# =========================
def get_unused_functions(test_dir):
    print("Running vulture...")
    output = run_cmd(["vulture", test_dir, "--min-confidence", "100"])
    unused = {}

    for line in output.splitlines():
        match = re.search(r"(.+):\d+: unused function '(\w+)'", line)
        if match:
            file, func = match.groups()
            unused.setdefault(file, set()).add(func)

    return unused


# =========================
# Step 4: Remove unused functions
# =========================
def make_unused_func_predicate(funcs):
    def should_remove(node):
        # Conservative strategy: keep functions with decorators
        if node.decorator_list:
            return False
        if node.name in funcs:
            print(f"[REMOVE FUNC] {node.name}")
            return True
        return False

    return should_remove


# =========================
# Step 5: Cross-file unused symbol detection
# =========================
def _collect_top_level_defs(filepath):
    """Return a dict of {name: has_decorator} for top-level functions/methods in a file."""
    with open(filepath, "r") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return {}

    defs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_decorated = bool(node.decorator_list)
            is_fixture = (
                any(
                    (
                        isinstance(d, ast.Call)
                        and isinstance(d.func, ast.Attribute)
                        and d.func.attr == "fixture"
                    )
                    or (isinstance(d, ast.Attribute) and d.attr == "fixture")
                    or (isinstance(d, ast.Name) and d.id == "fixture")
                    for d in node.decorator_list
                )
                if is_decorated
                else False
            )
            defs[node.name] = {"decorated": is_decorated, "fixture": is_fixture}
    return defs


def _read_file_text(filepath):
    with open(filepath, "r") as f:
        return f.read()


def _is_symbol_check_target(fpath, test_dir):
    relpath = os.path.relpath(fpath, test_dir)
    return relpath in {"common.py", "utils/timeout_manager.py"}


def find_cross_file_unused(test_dir, ignore_dirs, ignore_files):
    """Find functions/methods defined in one file but not referenced in others."""
    all_py_files = collect_py_files(test_dir, ignore_dirs, ignore_files)
    file_texts = {fpath: _read_file_text(fpath) for fpath in all_py_files}
    file_defs = {fpath: _collect_top_level_defs(fpath) for fpath in all_py_files}

    unused = defaultdict(list)
    for fpath, defs in file_defs.items():
        for name, info in defs.items():
            if name.startswith("test_"):
                continue
            if name.startswith("_"):
                continue
            if name in PYTEST_HOOKS:
                continue
            if "setup" in name or "teardown" in name:
                continue
            if info["fixture"]:
                continue

            referenced_elsewhere = False
            for other_path, text in file_texts.items():
                if other_path == fpath:
                    continue
                if re.search(rf"\b{re.escape(name)}\b", text):
                    referenced_elsewhere = True
                    break

            if not referenced_elsewhere:
                unused[fpath].append(name)

    return dict(unused)


# =========================
# Generic file processing
# =========================
def process_file(filepath, should_remove, dry_run=False):
    """Remove functions matching should_remove from filepath, preserving all formatting."""
    with open(filepath, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return

    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if should_remove(node):
                start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                ranges.append((start, node.end_lineno))

    if not ranges or dry_run:
        return

    lines = source.splitlines(keepends=True)
    to_remove = set()
    for start, end in ranges:
        for i in range(start - 1, end):
            to_remove.add(i)

    with open(filepath, "w") as f:
        f.write("".join(line for i, line in enumerate(lines) if i not in to_remove))


# =========================
# Main workflow
# =========================
def main(
    test_lists_dir, test_dir, ignore_dirs, ignore_files, symbol_check_files=None, dry_run=False
):
    # Step 1
    used_tests = get_used_tests(test_lists_dir)
    print(f"Found {len(used_tests)} unique test names from test lists")
    if ignore_dirs:
        print(f"Ignoring directories: {', '.join(sorted(ignore_dirs))}")
    if ignore_files:
        print(f"Ignoring files: {', '.join(sorted(ignore_files))}")

    # Step 2
    print(f"\nCleaning unused test cases in: {test_dir}")
    for path in collect_py_files(test_dir, ignore_dirs, ignore_files):
        # Only prune test_* cases from test modules.
        if not os.path.basename(path).startswith("test_"):
            continue
        process_file(path, make_unused_test_predicate(used_tests, path, test_dir), dry_run)

    # Step 3
    unused_funcs = get_unused_functions(test_dir)

    # Step 4
    print("\nCleaning unused helper functions...")
    for file, funcs in unused_funcs.items():
        if not os.path.exists(file):
            continue
        process_file(file, make_unused_func_predicate(funcs), dry_run)

    # Step 5
    print("\nCross-file unused symbol check...")
    if symbol_check_files:
        print(f"Only checking: {', '.join(sorted(symbol_check_files))}")
    cross_unused = find_cross_file_unused(test_dir, ignore_dirs, ignore_files)
    if cross_unused:
        for filepath, names in sorted(cross_unused.items()):
            rel = os.path.relpath(filepath, test_dir)
            for name in sorted(names):
                print(f"[UNUSED SYMBOL] {rel}: {name}")
        total = sum(len(v) for v in cross_unused.values())
        print(f"\nFound {total} symbols only referenced in their own file")
    else:
        print("No cross-file unused symbols found.")

    print("\nDone.")
    if dry_run:
        print("⚠️ This was a dry-run. No files were modified.")


# usage: python scripts/clean_tests.py \
#   --test-lists-dir=tests/integration/test_lists/ \
#   --test-dir=tests/integration/defs/ --dry-run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-lists-dir",
        required=True,
        help="Path to test_lists directory containing .txt/.yml files",
    )
    parser.add_argument(
        "--test-dir",
        default=DEFAULT_TEST_DIR,
        help="Path to test source directory to clean (default: tests)",
    )
    parser.add_argument(
        "--ignore-dirs",
        nargs="*",
        default=list(DEFAULT_IGNORE_DIRS),
        help=f"Directory names to skip (default: {' '.join(DEFAULT_IGNORE_DIRS)})",
    )
    parser.add_argument(
        "--ignore-files",
        nargs="*",
        default=list(DEFAULT_IGNORE_FILES),
        help=f"File names to skip (default: {' '.join(DEFAULT_IGNORE_FILES)})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    main(
        test_lists_dir=args.test_lists_dir,
        test_dir=args.test_dir,
        ignore_dirs=set(args.ignore_dirs),
        ignore_files=set(args.ignore_files),
        dry_run=args.dry_run,
    )
