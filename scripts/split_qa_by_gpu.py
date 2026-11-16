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
"""Split a QA list into 1–4, 5–8 and 9–16 GPU lists by reading source markers.

    python scripts/split_qa_by_gpu.py --test-list \
        tests/integration/test_lists/qa/llm_function_core.txt --out-dir out/

Unmarked cases require one GPU. GPU demand is the larger of the closest
``skip_less_device`` and ``skip_less_mpi_world_size`` bounds, assuming one MPI
rank per GPU. The device bound still requires that many GPUs visible to one
process; an MPI bound can span nodes when launched with that world size, or
falls back to visible GPUs when the world size is one. These lists classify
GPU counts only, not architecture, memory or launch topology.

Reads Python ASTs without importing tests, pytest, torch or TensorRT-LLM.
Preserves test-list suffixes, including TIMEOUT, and input order. A selector
covering multiple GPU buckets is expanded into its methods/parameter IDs.
Unclassifiable entries are skipped and recorded in ``*_skipped.jsonl``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import check_test_list as test_lists

_GPU_MARKERS = {"skip_less_device", "skip_less_mpi_world_size"}
_BUCKETS = ((1, 4), (5, 8), (9, 16))
_TEST_BASE = Path(__file__).resolve().parents[1] / "tests/integration/defs"
_Function = ast.FunctionDef | ast.AsyncFunctionDef


@lru_cache(maxsize=None)
def _source(path: Path) -> tuple[ast.Module, dict[str, ast.expr]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = test_lists.build_ast_index(str(path))[-1]
    return tree, constants


def _gpu_marks(expressions: list[ast.expr], constants: dict[str, ast.expr]) -> dict[str, int]:
    """Read marks in closest-first order, ignoring marks unrelated to GPU counts."""
    bounds: dict[str, int] = {}
    for expression in expressions:
        expression = test_lists._resolve_const_node(expression, constants)
        if isinstance(expression, (ast.List, ast.Tuple)):
            values = _gpu_marks(expression.elts, constants)
        elif (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in _GPU_MARKERS
        ):
            name = expression.func.attr
            if not expression.args:
                raise ValueError(f"{name} needs a positive integer argument")
            value = ast.literal_eval(test_lists._resolve_const_node(expression.args[0], constants))
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} needs a positive integer, got {value!r}")
            values = {name: value}
        else:
            continue
        for name, value in values.items():
            bounds.setdefault(name, value)
    return bounds


def _class_chain(node: ast.ClassDef, classes: dict[str, ast.ClassDef]) -> list[ast.ClassDef]:
    """Read local classes in reversed MRO, as pytest reads inherited class marks."""
    bases = [
        classes[base.id] for base in node.bases if isinstance(base, ast.Name) and base.id in classes
    ]
    sequences = [list(reversed(_class_chain(base, classes))) for base in bases] + [list(bases)]
    order = [node]
    while any(sequences):
        sequences = [sequence for sequence in sequences if sequence]
        candidate = next(
            (
                sequence[0]
                for sequence in sequences
                if all(sequence[0] not in other[1:] for other in sequences)
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"cannot resolve local class inheritance for {node.name}")
        order.append(candidate)
        for sequence in sequences:
            if sequence[0] is candidate:
                sequence.pop(0)
    return list(reversed(order))


def _function_cases(
    function: _Function,
    class_decorators: list[ast.expr],
    enclosing: dict[str, int],
    constants: dict[str, ast.expr],
    param_id: str | None,
) -> list[tuple[str | None, int]]:
    """Return parameter IDs and GPU demands, resolving IDs only for varying bounds."""
    function_marks = _gpu_marks(list(reversed(function.decorator_list)), constants)
    decorators = [*reversed(function.decorator_list), *reversed(class_decorators)]
    groups = []
    for decorator in decorators:
        if not test_lists._is_parametrize_call(decorator):
            continue
        if len(decorator.args) < 2:
            raise ValueError("parametrize needs positional argnames and argvalues")
        rows = test_lists._argvalues_elts(decorator.args[1], constants)
        if rows is None:
            raise ValueError("cannot statically read parametrize argvalues")
        row_marks = []
        for row in rows:
            marks = []
            if test_lists._is_pytest_param(row):
                marks = [keyword.value for keyword in row.keywords if keyword.arg == "marks"]
            values = _gpu_marks(marks, constants)
            # A function's own marks precede its pytest.param marks on the same item.
            row_marks.append(
                {name: value for name, value in values.items() if name not in function_marks}
            )
        if not row_marks:
            raise ValueError("parametrize has no rows")
        groups.append((decorator, row_marks))

    def demand(parameter_marks: list[dict[str, int]]) -> int:
        bounds: dict[str, int] = {}
        for marks in parameter_marks:
            for name, value in marks.items():
                bounds.setdefault(name, value)
        return max({**enclosing, **bounds, **function_marks}.values(), default=1)

    if all(all(marks == rows[0] for marks in rows) for _, rows in groups):
        return [(param_id, demand([rows[0] for _, rows in groups]))]

    choices = []
    for decorator, rows in groups:
        ids = test_lists._get_parametrize_ids(decorator, constants)
        if ids is None or len(ids) != len(rows):
            raise ValueError("cannot resolve parameter IDs with different GPU requirements")
        choices.append(list(zip(ids, rows)))
    cases = [
        ("-".join(row[0] for row in combo), demand([row[1] for row in combo]))
        for combo in product(*choices)
    ]
    ids = [case[0] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate parameter IDs with different GPU requirements")
    if param_id is not None:
        cases = [case for case in cases if case[0] == param_id]
        if not cases:
            raise ValueError(f"parameter ID {param_id!r} not found")
    return cases


def _selector_cases(selector: str, test_base: Path) -> list[tuple[str, int]]:
    entry = test_lists.parse_test_entry(selector)
    if entry is None or entry[0] == "MALFORMED":
        raise ValueError(f"invalid test selector {selector!r}")
    relative, class_name, function_name, param_id = entry
    resolved = test_lists._resolve_path(relative, str(test_base))
    if resolved is None:
        raise ValueError(f"test source {relative!r} not found")
    tree, constants = _source(Path(resolved))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    module_marks = (
        _gpu_marks([constants["pytestmark"]], constants) if "pytestmark" in constants else {}
    )
    enclosing = dict(module_marks)
    class_decorators: list[ast.expr] = []
    if class_name is None and function_name in classes:
        class_name, function_name = function_name, None
    if class_name is not None:
        if class_name not in classes:
            raise ValueError(f"class {class_name!r} not found")
        nodes = []
        class_marks: dict[str, int] = {}
        for cls in _class_chain(classes[class_name], classes):
            own_marks = [
                statement.value
                for statement in cls.body
                if isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "pytestmark"
                    for target in statement.targets
                )
            ]
            for name, value in _gpu_marks(
                [*own_marks, *reversed(cls.decorator_list)], constants
            ).items():
                class_marks.setdefault(name, value)
            class_decorators = [*cls.decorator_list, *class_decorators]
            nodes.extend(cls.body)
        enclosing.update(class_marks)
    else:
        nodes = tree.body
    functions = {node.name: node for node in nodes if isinstance(node, _Function)}
    if function_name is None:
        functions = {name: node for name, node in functions.items() if name.startswith("test")}
    elif function_name in functions:
        functions = {function_name: functions[function_name]}
    else:
        raise ValueError(f"function {function_name!r} not found")
    if not functions:
        raise ValueError("selector contains no test functions")

    cases = []
    base = selector.split("[", 1)[0]
    for name, function in functions.items():
        prefix = f"{base}::{name}" if function_name is None else base
        for identifier, required in _function_cases(
            function, class_decorators, enclosing, constants, param_id
        ):
            case = f"{prefix}[{identifier}]" if identifier is not None else prefix
            cases.append((case, required))
    if len({_bucket(required) for _, required in cases}) == 1:
        return [(selector, max(required for _, required in cases))]
    return cases


def _bucket(required: int) -> tuple[int, int]:
    for lower, upper in _BUCKETS:
        if required <= upper:
            return lower, upper
    raise ValueError(f"requires {required} GPUs, above the supported maximum of 16")


def split_test_list(
    test_list: Path, out_dir: Path, test_base: Path = _TEST_BASE
) -> dict[Path, int]:
    """Write three disjoint txt lists and a report of unclassifiable entries."""
    buckets: dict[tuple[int, int], list[str]] = {bounds: [] for bounds in _BUCKETS}
    skipped = []
    for lineno, original in enumerate(test_list.read_text(encoding="utf-8").splitlines(), start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        selector, suffix = parts[0], parts[1] if len(parts) > 1 else ""
        try:
            cases = _selector_cases(selector, test_base)
            classified = [
                (_bucket(required), f"{case} {suffix}" if suffix else case)
                for case, required in cases
            ]
        except (TypeError, ValueError, SyntaxError) as error:
            skipped.append({"line": lineno, "entry": original, "reason": str(error)})
            print(f"{test_list}:{lineno}: skipped {selector}: {error}", file=sys.stderr)
            continue
        for bucket, case in classified:
            buckets[bucket].append(case)
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = out_dir / f"{test_list.stem}_skipped.jsonl"
    skipped_path.write_text(
        "".join(f"{json.dumps(entry, ensure_ascii=False)}\n" for entry in skipped), encoding="utf-8"
    )
    if skipped:
        print(f"Skipped {len(skipped)} entries; see {skipped_path}", file=sys.stderr)
    counts = {}
    for (lower, upper), lines in buckets.items():
        path = out_dir / f"{test_list.stem}_{lower}to{upper}gpu.txt"
        header = (
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            f"# GPU marker demand {lower}–{upper}; unmarked cases default to one GPU.\n"
            "# MPI demand assumes one rank per GPU; device demand remains per process.\n\n"
        )
        text = header + "".join(f"{line}\n" for line in lines)
        path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        counts[path] = len(lines)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        counts = split_test_list(args.test_list, args.out_dir)
    except (OSError, ValueError, SyntaxError) as error:
        parser.error(str(error))
    for path, count in counts.items():
        print(f"{path}: {count} cases")
    print(f"Total: {sum(counts.values())} cases")


if __name__ == "__main__":
    main()
