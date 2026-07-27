#!/usr/bin/env python3
"""Verify test lists for L0, QA, and waives file; AST-based validation via --validate.

This script is used to verify test lists for L0, QA, and waives file.
Also provides AST-based static validation of test list entries (--validate flag).

Usage:
When in a development or container environment, run the following command:
    python $LLM_ROOT/scripts/check_test_list.py --l0 --qa --waive

For AST-based validation (no pytest or model weights needed):
    python $LLM_ROOT/scripts/check_test_list.py --validate

Options:
--l0:       Check only the L0 tests located in $LLM_ROOT/tests/integration/test_list/test_db/*.yml.
--qa:       Check only the QA tests under $LLM_ROOT/tests/integration/test_list/*.txt.
--waive:    Check only the tests in $LLM_ROOT/tests/integration/test_list/waives.txt.
--validate: Run AST-based validation of test list entries against source files.

Note:
All the perf tests will be excluded since they are generated dynamically.
"""
import argparse
import ast
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

# The markers in our test lists, need to be preprocess before checking
MARKER_LIST_IN_TEST = [" TIMEOUT"]

# AST validation defaults
_DEFAULT_TEST_LISTS_DIR = "tests/integration/test_lists"
_DEFAULT_TEST_BASE_DIR = "tests/integration/defs"
# Paths whose tests are generated dynamically — skip AST validation
_EXCLUDED_PATH_PREFIXES = ("perf/", )

# =============================================================================
# AST-based test list validation
# =============================================================================


def parse_test_entry(line: str):
    """Return (rel_path, class_name_or_None, func_name, param_id_or_None) from a line.

    Handles .txt and .yml formats, and waives.txt's 'full:GPU/path' prefix.
    Returns None for non-test lines.
    Returns ("MALFORMED", original_token, None, None) for structurally broken entries.
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
    file_path = parts[0]
    # Strip 'full:GPU/' or similar hardware-qualifier prefix (e.g. 'full:GH200/...')
    if ":" in file_path and "/" in file_path:
        slash_idx = file_path.index("/")
        file_path = file_path[slash_idx + 1:]
    if not file_path.endswith(".py"):
        return None
    if len(parts) == 2:
        class_name, func_part = None, parts[1]
    elif len(parts) >= 3:
        class_name, func_part = parts[1], parts[2]
    else:
        return None
    param_id = None
    if "[" in func_part:
        bracket = func_part.index("[")
        param_id = (func_part[bracket + 1:func_part.rindex("]")]
                    if "]" in func_part else None)
        func_part = func_part[:bracket]
    if not func_part:
        return "MALFORMED", token, None, None
    return file_path, class_name, func_part, param_id


def _is_parametrize_call(node) -> bool:
    """Return True if node is @pytest.mark.parametrize(...) or @parametrize_with_ids(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "parametrize":
        return True
    if isinstance(func, ast.Name) and func.id == "parametrize_with_ids":
        return True
    return False


def _ast_constant_str(node) -> str | None:
    """Return str() of an AST constant node, or None if not a simple literal."""
    if isinstance(node, ast.Constant) and isinstance(
            node.value, (str, int, float, bool, type(None))):
        return str(node.value)
    return None


def _get_parametrize_with_ids_ids(call: ast.Call) -> list[str] | None:
    """Extract IDs from a parametrize_with_ids(argnames, argvalues) call.

    parametrize_with_ids generates IDs like "argname=value" joined with "-".
    """
    if len(call.args) < 2:
        return None
    argnames_node, argvalues_node = call.args[0], call.args[1]

    if isinstance(argnames_node, ast.Constant) and isinstance(
            argnames_node.value, str):
        argname_list = [n.strip() for n in argnames_node.value.split(",")]
    elif isinstance(argnames_node, (ast.List, ast.Tuple)):
        argname_list = []
        for elt in argnames_node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                argname_list.append(elt.value.strip())
            else:
                return None
    else:
        return None

    if not isinstance(argvalues_node, ast.List):
        return None

    ids = []
    for elt in argvalues_node.elts:
        val_str = _ast_constant_str(elt)
        if val_str is not None:
            if len(argname_list) != 1:
                return None
            ids.append(f"{argname_list[0]}={val_str}")
        elif isinstance(elt, (ast.Tuple, ast.List)):
            if len(elt.elts) != len(argname_list):
                return None
            parts = []
            for name, val_node in zip(argname_list, elt.elts):
                v = _ast_constant_str(val_node)
                if v is None:
                    return None
                parts.append(f"{name}={v}")
            ids.append("-".join(parts))
        else:
            return None
    return ids


def _get_parametrize_ids(call: ast.Call) -> list[str] | None:
    """Extract the list of IDs from a parametrize call.

    Handles both pytest.mark.parametrize and parametrize_with_ids.
    Tries ids= kwarg first, then falls back to string/int literal argvalues.
    Returns None if IDs cannot be determined statically.
    """
    func = call.func
    if isinstance(func, ast.Name) and func.id == "parametrize_with_ids":
        return _get_parametrize_with_ids_ids(call)

    for kw in call.keywords:
        if kw.arg == "ids":
            if not isinstance(kw.value, ast.List):
                return None  # ids= is a lambda/variable — indeterminate
            ids = []
            for elt in kw.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    ids.append(elt.value)
                else:
                    return None
            return ids

    # No ids= — use string/int literal argvalues (or pytest.param(...))
    if len(call.args) < 2:
        return None
    argvalues = call.args[1]
    if isinstance(argvalues, ast.List):
        ids = []
        for elt in argvalues.elts:
            val = _ast_constant_str(elt)
            if val is not None:
                ids.append(val)
            elif isinstance(elt, ast.Call) and _is_pytest_param(elt):
                inner = _pytest_param_id(elt)
                if inner is None:
                    return None
                ids.append(inner)
            else:
                return None
        return ids
    return None


def _is_pytest_param(node) -> bool:
    """Return True if node is a pytest.param(...) call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "param"


def _pytest_param_id(call: ast.Call) -> str | None:
    """Return the ID string for a pytest.param(v1, v2, ...) call.

    Uses the explicit id= kwarg if present; otherwise pytest joins all
    positional arg values with '-' (matching pytest's own ID generation).
    Returns None if the ID cannot be determined statically.
    """
    for kw in call.keywords:
        if kw.arg == "id":
            return _ast_constant_str(kw.value)
    parts = []
    for arg in call.args:
        v = _ast_constant_str(arg)
        if v is None:
            return None
        parts.append(v)
    return "-".join(parts) if parts else None


def _compute_valid_param_ids(func_node,
                             class_decorators=None) -> set[str] | None:
    """Return the set of valid parametrize IDs for a function node.

    Iterates method-level parametrize decorators in reverse source order
    (innermost first) then appends class-level parametrize groups at the end.
    Returns None if any parametrize has indeterminate IDs.
    Returns empty set if there are no parametrize decorators.
    """
    groups = []
    for decorator in reversed(func_node.decorator_list):
        if not _is_parametrize_call(decorator):
            continue
        ids = _get_parametrize_ids(decorator)
        if ids is None:
            return None
        groups.append(ids)
    for decorator in reversed(class_decorators or []):
        if not _is_parametrize_call(decorator):
            continue
        ids = _get_parametrize_ids(decorator)
        if ids is None:
            return None
        groups.append(ids)
    if not groups:
        return set()
    return {"-".join(combo) for combo in product(*groups)}


def build_ast_index(filepath: str):
    """Return (classes, top_level_funcs, top_level_nodes, class_method_nodes).

    classes: {class_name: {'methods': set[str], 'bases': list[str],
                            'decorators': list[ast.Call]}}
    Returns (None, None, None, None) on error.
    """
    try:
        source = Path(filepath).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filepath)
    except (OSError, SyntaxError):
        return None, None, None, None

    classes = {}
    class_method_nodes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {}
            for n in node.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[n.name] = n
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            class_decorators = [
                d for d in node.decorator_list if _is_parametrize_call(d)
            ]
            classes[node.name] = {
                "methods": set(methods),
                "bases": bases,
                "decorators": class_decorators,
            }
            class_method_nodes[node.name] = methods

    top_level_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    top_level_funcs = set(top_level_nodes)
    return classes, top_level_funcs, top_level_nodes, class_method_nodes


def _matches_parameterized(func_name: str, methods: set) -> bool:
    """Return True if func_name is a @parameterized.expand expansion of any method."""
    for method in methods:
        if func_name.startswith(method + "_"):
            suffix = func_name[len(method) + 1:]
            if re.match(r"^\d+(_|$)", suffix):
                return True
    return False


def _has_method(class_name: str, method_name: str, ast_cache: dict,
                visited: set) -> bool:
    """Check if class_name defines or inherits method_name."""
    if class_name in visited:
        return False
    visited.add(class_name)

    for classes, _, _, _ in ast_cache.values():
        if not classes or class_name not in classes:
            continue
        info = classes[class_name]
        if method_name in info["methods"]:
            return True
        if _matches_parameterized(method_name, info["methods"]):
            return True
        for base in info["bases"]:
            if _has_method(base, method_name, ast_cache, visited):
                return True

    return False


def collect_entries(test_lists_dir: str, include_waives: bool = False):
    """Walk test_lists_dir and return ({entry_tuple: [refs]}, [malformed_refs])."""
    entries = defaultdict(list)
    malformed = []
    for root, _, files in os.walk(test_lists_dir):
        for fname in sorted(files):
            if fname == "waives.txt" and not include_waives:
                continue
            if not (fname.endswith(".txt") or fname.endswith(".yml")):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    entry = parse_test_entry(line)
                    if entry is None:
                        continue
                    if entry[0] == "MALFORMED":
                        malformed.append(f"{fpath}:{lineno}: {entry[1]}")
                    else:
                        entries[entry].append(f"{fpath}:{lineno}")
    return entries, malformed


def _resolve_path(rel_path: str, test_base_dir: str) -> str | None:
    """Try to resolve rel_path to an existing absolute path."""
    candidate = os.path.join(test_base_dir, rel_path)
    if os.path.exists(candidate):
        return candidate
    tests_dir = os.path.dirname(os.path.dirname(test_base_dir))
    candidate2 = os.path.join(tests_dir, rel_path)
    if os.path.exists(candidate2):
        return candidate2
    return None


def _ensure_dir_indexed(abs_path: str, ast_cache: dict) -> None:
    """Index all .py files in the same directory as abs_path into ast_cache."""
    directory = os.path.dirname(abs_path)
    for fname in os.listdir(directory):
        if not fname.endswith(".py"):
            continue
        sibling = os.path.join(directory, fname)
        if sibling not in ast_cache:
            ast_cache[sibling] = build_ast_index(sibling)


def validate_test_lists(test_lists_dir: str, test_base_dir: str):
    """Return list of error strings from AST-based validation."""
    active_entries, active_malformed = collect_entries(test_lists_dir,
                                                       include_waives=False)
    waive_entries, waive_malformed = collect_entries(test_lists_dir,
                                                     include_waives=True)
    all_entries = dict(active_entries)
    for key, refs in waive_entries.items():
        if key not in all_entries:
            all_entries[key] = refs

    active_base_index: dict = defaultdict(set)
    for rel_path, class_name, func_name, param_id in active_entries:
        active_base_index[(rel_path, class_name, func_name)].add(param_id)

    ast_cache = {}
    errors = []

    active_malformed_set = set(active_malformed)
    for ref in active_malformed + [
            r for r in waive_malformed if r not in active_malformed_set
    ]:
        errors.append(f"MALFORMED ENTRY: {ref}")

    # Flag waive IDs that are a strict prefix of an active ID (likely truncated).
    waive_only_keys = {k for k in waive_entries if k not in active_entries}
    for rel_path, class_name, func_name, param_id in sorted(
            waive_only_keys, key=lambda x:
        (x[0], x[1] or "", x[2], x[3] or "")):
        if param_id is None:
            continue
        base = (rel_path, class_name, func_name)
        active_ids = active_base_index.get(base, set())
        prefix = param_id + "-"
        matches = [i for i in active_ids if i and i.startswith(prefix)]
        if matches:
            refs = waive_entries[(rel_path, class_name, func_name, param_id)]
            errors.append(
                "TRUNCATED WAIVE ID: {}{}::{}\n"
                "  Waived ID '{}' looks truncated; matching active IDs: {}\n{}".
                format(
                    rel_path,
                    f"::{class_name}" if class_name else "",
                    func_name,
                    param_id,
                    sorted(matches)[:8],
                    "\n".join(f"  -> {r}" for r in refs[:3]),
                ))

    for (rel_path, class_name, func_name,
         param_id), refs in sorted(all_entries.items(),
                                   key=lambda x:
                                   (x[0][0], x[0][1] or "", x[0][2])):
        if any(rel_path.startswith(p) for p in _EXCLUDED_PATH_PREFIXES):
            continue

        is_waive_only = (rel_path, class_name, func_name,
                         param_id) not in active_entries
        abs_path = _resolve_path(rel_path, test_base_dir)

        if abs_path is None:
            # Always flag missing files — a waive for a deleted file is stale.
            errors.append("FILE NOT FOUND: {}{}\n{}".format(
                rel_path,
                " (waive-only — stale waive?)" if is_waive_only else "",
                "\n".join(f"  -> {r}" for r in refs[:3])))
            continue

        if abs_path not in ast_cache:
            ast_cache[abs_path] = build_ast_index(abs_path)

        classes, top_level, top_level_nodes, class_method_nodes = (
            ast_cache[abs_path])
        if classes is None:
            errors.append(f"PARSE ERROR: {rel_path}")
            continue

        func_node = None

        if class_name:
            if class_name not in classes:
                # Always flag missing classes — a waive for a deleted class is stale.
                errors.append("CLASS NOT FOUND: {}::{}{}\n{}".format(
                    rel_path, class_name,
                    " (waive-only — stale waive?)" if is_waive_only else "",
                    "\n".join(f"  -> {r}" for r in refs[:3])))
                continue
            direct_methods = classes[class_name]["methods"]
            if func_name not in direct_methods and not _matches_parameterized(
                    func_name, direct_methods):
                _ensure_dir_indexed(abs_path, ast_cache)
                if not _has_method(class_name, func_name, ast_cache, set()):
                    errors.append("METHOD NOT FOUND: {}::{}::{}{}\n{}".format(
                        rel_path, class_name, func_name,
                        " (waive-only — stale waive?)" if is_waive_only else "",
                        "\n".join(f"  -> {r}" for r in refs[:3])))
                    continue
            func_node = class_method_nodes.get(class_name, {}).get(func_name)
        else:
            if func_name not in top_level and func_name not in classes:
                # Always flag missing functions — a waive for a deleted function is stale.
                errors.append("FUNCTION NOT FOUND: {}::{}{}\n{}".format(
                    rel_path, func_name,
                    " (waive-only — stale waive?)" if is_waive_only else "",
                    "\n".join(f"  -> {r}" for r in refs[:3])))
                continue
            if (func_name not in top_level and func_name in classes
                    and param_id is not None):
                errors.append(
                    "CLASS USED WITHOUT METHOD: {}::{}\n"
                    "  Entry has param_id '{}' but no method specified\n{}".
                    format(rel_path, func_name, param_id,
                           "\n".join(f"  -> {r}" for r in refs[:3])))
                continue
            func_node = top_level_nodes.get(func_name)

        if param_id and func_node:
            class_decs = (classes[class_name]["decorators"]
                          if class_name and class_name in classes else None)
            valid_ids = _compute_valid_param_ids(func_node, class_decs)
            if valid_ids is not None and len(valid_ids) > 0:
                if param_id not in valid_ids:
                    errors.append("INVALID PARAMETRIZE ID: {}{}::{}\n"
                                  "  ID '{}' not in valid IDs: {}\n{}".format(
                                      rel_path,
                                      f"::{class_name}" if class_name else "",
                                      func_name,
                                      param_id,
                                      sorted(valid_ids)[:8],
                                      "\n".join(f"  -> {r}" for r in refs[:3]),
                                  ))

        if is_waive_only and not rel_path.startswith("unittest/"):
            param_suffix = f"[{param_id}]" if param_id else ""
            errors.append(
                "WAIVE NOT IN ACTIVE TEST LISTS: {}{}::{}{}\n{}".format(
                    rel_path,
                    f"::{class_name}" if class_name else "",
                    func_name,
                    param_suffix,
                    "\n".join(f"  -> {r}" for r in refs[:3]),
                ))

    return errors


# =============================================================================
# L0 / QA / Waive verification (runtime, requires pytest + model weights)
# =============================================================================


def install_python_dependencies(llm_src):
    subprocess.run(f"cd {llm_src} && pip3 install -r requirements-dev.txt",
                   shell=True,
                   check=True)
    subprocess.run(
        f"pip3 install --force-reinstall --no-deps {llm_src}/../tensorrt_llm-*.whl",
        shell=True,
        check=True)
    subprocess.run(
        "pip3 install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-tensorrt-pypi/simple "
        "--ignore-installed trt-test-db==1.8.5+bc6df7",
        shell=True,
        check=True)


def verify_l0_test_lists(llm_src):
    test_db_path = f"{llm_src}/tests/integration/test_lists/test-db"
    test_list = f"{llm_src}/l0_test.txt"

    # Remove dynamically generated perf tests
    # Exclude perf_sanity tests from being removed since they are different and statically defined
    for file_path in glob.glob(os.path.join(test_db_path, "*perf*")):
        if "perf_sanity" not in os.path.basename(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
    subprocess.run(
        f"trt-test-db -d {test_db_path} --test-names --output {test_list}",
        shell=True,
        check=True)

    # Remove the duplicated test names
    cleaned_lines = set()
    with open(test_list, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Remove markers and rest of the line if present
        cleaned_line = line.strip()

        # Handle ISOLATION marker removal (including comma patterns)
        if 'ISOLATION,' in cleaned_line:
            # Case: "ISOLATION,OTHER_MARKER" -> remove "ISOLATION,"
            cleaned_line = cleaned_line.replace('ISOLATION,', '').strip()
        elif ',ISOLATION' in cleaned_line:
            # Case: "OTHER_MARKER,ISOLATION" -> remove ",ISOLATION"
            cleaned_line = cleaned_line.replace(',ISOLATION', '').strip()
        elif ' ISOLATION' in cleaned_line:
            # Case: standalone "ISOLATION" -> remove " ISOLATION"
            cleaned_line = cleaned_line.replace(' ISOLATION', '').strip()

        # Handle other markers (like TIMEOUT) - remove marker and everything after it
        for marker in MARKER_LIST_IN_TEST:
            if marker in cleaned_line and marker != " ISOLATION":
                cleaned_line = cleaned_line.split(marker, 1)[0].strip()
                break

        if cleaned_line:
            cleaned_lines.add(cleaned_line)

    with open(test_list, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(cleaned_lines))

    subprocess.run(
        f"cd {llm_src}/tests/integration/defs && "
        f"pytest --test-list={test_list} --output-dir={llm_src} -s --co -q",
        shell=True,
        check=True)


def verify_qa_test_lists(llm_src):
    test_qa_path = f"{llm_src}/tests/integration/test_lists/qa"
    # Remove dynamically generated perf tests
    subprocess.run(f"rm -f {test_qa_path}/*perf*", shell=True, check=True)
    test_def_files = subprocess.check_output(
        f"ls -d {test_qa_path}/*.txt", shell=True).decode().strip().split('\n')
    for test_def_file in test_def_files:
        subprocess.run(
            f"cd {llm_src}/tests/integration/defs && "
            f"pytest --test-list={test_def_file} --output-dir={llm_src} -s --co -q",
            shell=True,
            check=True)
        # append all the test_def_file to qa_test.txt
        with open(f"{llm_src}/qa_test.txt", "a") as f:
            with open(test_def_file, "r") as test_file:
                lines = test_file.readlines()
                for line in lines:
                    # Remove 'TIMEOUT' marker and strip spaces
                    cleaned_line = line.split(" TIMEOUT ", 1)[0].strip()
                    if cleaned_line:
                        f.write(f"{cleaned_line}\n")


def check_waive_duplicates(llm_src):
    """Check for duplicate entries in waives.txt and write report."""
    waives_list_path = f"{llm_src}/tests/integration/test_lists/waives.txt"
    dup_cases_record = f"{llm_src}/dup_cases.txt"

    # Track all occurrences: processed_line -> [(line_no, original_line), ...]
    dedup_lines = {}

    with open(waives_list_path, "r") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, 1):
        original_line = line.strip()
        line = line.strip()

        if not line:
            continue

        # Check for SKIP marker in waives.txt and split by the first occurrence
        line = line.split(" SKIP", 1)[0].strip()

        # Track all occurrences of each processed line
        if line in dedup_lines:
            dedup_lines[line].append((line_no, original_line))
        else:
            dedup_lines[line] = [(line_no, original_line)]

    # Write duplicate report after processing all lines
    for processed_line, occurrences in dedup_lines.items():
        if len(occurrences) > 1:
            with open(dup_cases_record, "a") as f:
                f.write(
                    f"Duplicate waive records found for '{processed_line}' ({len(occurrences)} occurrences):\n"
                )
                for i, (line_no, original_line) in enumerate(occurrences, 1):
                    f.write(
                        f"  Occurrence {i} at line {line_no}: '{original_line}'\n"
                    )
                f.write(f"\n")


def verify_waive_list(llm_src, args):
    waives_list_path = f"{llm_src}/tests/integration/test_lists/waives.txt"
    non_existent_cases_record = f"{llm_src}/nonexits_cases.json"

    processed_lines = set()
    with open(waives_list_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Skip Perf tests due to they are dynamically generated
        if "perf/test_perf.py" in line:
            continue

        # Check for SKIP marker in waives.txt and split by the first occurrence
        line = line.split(" SKIP", 1)[0].strip()

        # If the line starts with 'full:', process it
        if line.startswith("full:"):
            line = line.split("/", 1)[1].lstrip("/")

        # Skip unittests due to we don't need to have an entry in test-db yml
        if line.startswith("unittest/"):
            continue

        # Check waived cases also in l0_text.txt and qa_text.txt
        found_in_l0_qa = False
        if args.l0:
            with open(f"{llm_src}/l0_test.txt", "r") as f:
                l0_lines = f.readlines()
                for l0_line in l0_lines:
                    if line == l0_line.strip():
                        found_in_l0_qa = True
                        break
        if args.qa:
            with open(f"{llm_src}/qa_test.txt", "r") as f:
                qa_lines = f.readlines()
                for qa_line in qa_lines:
                    if line == qa_line.strip():
                        found_in_l0_qa = True
                        break
        if not found_in_l0_qa:
            with open(non_existent_cases_record, "a") as f:
                f.write(
                    f"Non-existent test name in l0 or qa list found in waives.txt: {line}\n"
                )

        processed_lines.add(line)

    # Write the processed lines to a tmp file
    tmp_waives_file = f"{llm_src}/processed_waive_list.txt"
    with open(tmp_waives_file, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(processed_lines))

    subprocess.run(
        f"cd {llm_src}/tests/integration/defs && "
        f"pytest --test-list={tmp_waives_file} --output-dir={llm_src} -s --co -q",
        shell=True,
        check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Check test lists for L0 and QA.")
    parser.add_argument("--l0",
                        action="store_true",
                        help="Enable L0 test list verification.")
    parser.add_argument("--qa",
                        action="store_true",
                        help="Enable QA test list verification.")
    parser.add_argument("--waive",
                        action="store_true",
                        help="Enable test list verification for waive file.")
    parser.add_argument(
        "--check-duplicate-waives",
        action="store_true",
        help="Enable duplicate check in waives.txt (fails if duplicates found)."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=
        "Run AST-based validation of test list entries against source files.")
    parser.add_argument(
        "--test-lists-dir",
        default=_DEFAULT_TEST_LISTS_DIR,
        help=
        f"Test lists directory for --validate (default: {_DEFAULT_TEST_LISTS_DIR})",
    )
    parser.add_argument(
        "--test-base-dir",
        default=_DEFAULT_TEST_BASE_DIR,
        help=
        f"Base directory for test source files for --validate (default: {_DEFAULT_TEST_BASE_DIR})",
    )
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    llm_src = os.path.abspath(os.path.join(script_dir, "../"))

    # Only skip installing dependencies if ONLY --check-duplicates or --validate is used
    if args.l0 or args.qa or args.waive:
        install_python_dependencies(llm_src)

    pass_flag = True
    # Verify L0 test lists
    if args.l0:
        print("-----------Starting L0 test list verification...-----------",
              flush=True)
        verify_l0_test_lists(llm_src)
    else:
        print("-----------Skipping L0 test list verification.-----------",
              flush=True)

    # Verify QA test lists
    if args.qa:
        print("-----------Starting QA test list verification...-----------",
              flush=True)
        verify_qa_test_lists(llm_src)
    else:
        print("-----------Skipping QA test list verification.-----------",
              flush=True)

    # Verify waive test lists
    if args.waive:
        print("-----------Starting waive list verification...-----------",
              flush=True)
        verify_waive_list(llm_src, args)
    else:
        print("-----------Skipping waive list verification.-----------",
              flush=True)

    # Check for duplicates in waives.txt if requested
    if args.check_duplicate_waives:
        print("-----------Checking for duplicates in waives.txt...-----------",
              flush=True)
        check_waive_duplicates(llm_src)

    # AST-based validation (no pytest or model weights needed)
    if args.validate:
        print("-----------Starting AST test list validation...-----------",
              flush=True)
        errors = validate_test_lists(args.test_lists_dir, args.test_base_dir)
        if errors:
            print(f"Found {len(errors)} validation error(s):\n",
                  file=sys.stderr)
            for err in errors:
                print(err, file=sys.stderr)
                print(file=sys.stderr)
            pass_flag = False
        else:
            entries, _ = collect_entries(args.test_lists_dir)
            print(f"OK: {len(entries)} unique test entries validated.")

    invalid_json_file = os.path.join(llm_src, "invalid_tests.json")
    if os.path.isfile(invalid_json_file) and os.path.getsize(
            invalid_json_file) > 0:
        print("Invalid cases:")
        with open(invalid_json_file, "r") as f:
            print(f.read())
        print("Invalid test names found, please correct them first!!!\n")
        pass_flag = False

    duplicate_cases_file = os.path.join(llm_src, "dup_cases.txt")
    if os.path.isfile(duplicate_cases_file) and os.path.getsize(
            duplicate_cases_file) > 0:
        print("Duplicate cases found:")
        with open(duplicate_cases_file, "r") as f:
            print(f.read())
        print(
            "Duplicate test names found in waives.txt, please delete one or combine them first!!!\n"
        )
        if args.check_duplicate_waives:
            pass_flag = False

    non_existent_cases_file = os.path.join(llm_src, "nonexits_cases.json")
    if os.path.isfile(non_existent_cases_file) and os.path.getsize(
            non_existent_cases_file) > 0:
        print("Non-existent cases found in waives.txt:")
        with open(non_existent_cases_file, "r") as f:
            print(f.read())
        print(
            "Non-unit test test name in waives.txt but not in l0 test list or qa list, please delete them first!!!\n"
        )
        pass_flag = False

    if not pass_flag:
        exit(1)


if __name__ == "__main__":
    main()
