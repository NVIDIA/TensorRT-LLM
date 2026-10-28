# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private-access lint for the serving frontends.

The HTTP server (tensorrt_llm/serve, without serve/scripts), the CLI
(tensorrt_llm/commands) and the gRPC frontends (tensorrt_llm/grpc) may use only
the public API of other TensorRT-LLM packages. This test counts every access to
a private member that belongs to another package and compares the result with
private_access_baseline.yaml, which lists the accesses that existed when the
lint was introduced. The baseline only shrinks.

Counting rule: one AST node per access. An access is an attribute
`<receiver>._name` or a getattr/hasattr/setattr/delattr call with the literal
"_name", where the receiver is not `self`, `cls` or `super()` and `_name` has one
leading underscore and is not a dunder. `a._b._c` is two accesses. Names
imported with `from m import _x` are not accesses. Ownership: OVERRIDES first,
then a name index over tensorrt_llm/**/*.py and *.pyi.

Run `python tests/unittest/serve/test_private_access_lint.py --report` to print
the current inventory.
"""

import ast
import collections
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_FILE = Path(__file__).resolve().with_name("private_access_baseline.yaml")
SCOPES = ("tensorrt_llm/serve/", "tensorrt_llm/commands/", "tensorrt_llm/grpc/")
EXCLUDED = ("tensorrt_llm/serve/scripts/", "tensorrt_llm/grpc/openengine/_generated/")
VISUAL_GEN_SCOPE = "tensorrt_llm/serve/api/visual_gen/"
OPENENGINE_SCOPE = "tensorrt_llm/grpc/openengine/"

# Lower this when you remove accesses; never raise it.
MAX_TOTAL = 50

# (path, member) -> (verdict, reason). Verdicts: "cross", "same-package", "third-party".
# Needed only where the name index cannot tell: the name is defined both inside and
# outside the scanning package, or a third-party name collides with a TensorRT-LLM name.
OVERRIDES = {
    ("tensorrt_llm/serve/_telemetry.py", "_serve"): ("third-party", "uvicorn.Server._serve"),
    ("tensorrt_llm/serve/_telemetry.py", "_captured_signals"): (
        "third-party",
        "uvicorn.Server._captured_signals",
    ),
    ("tensorrt_llm/serve/cluster_storage.py", "_loop"): ("third-party", "asyncio.Queue._loop"),
    ("tensorrt_llm/serve/render/fingerprint.py", "_mergeable_ranks"): (
        "third-party",
        "tiktoken Encoding",
    ),
    ("tensorrt_llm/serve/render/fingerprint.py", "_special_tokens"): (
        "third-party",
        "tiktoken Encoding",
    ),
    ("tensorrt_llm/serve/render/fingerprint.py", "_pat_str"): ("third-party", "tiktoken Encoding"),
    ("tensorrt_llm/serve/router.py", "_lock"): ("same-package", "serve/router.py Router._lock"),
    # Generic names of serve's own router objects: the name index classifies them
    # correctly today, but a definition of the same name in another package would
    # turn them into "unresolved" and fail unrelated pull requests.
    ("tensorrt_llm/serve/router.py", "_servers"): (
        "same-package",
        "serve/router.py Router._servers",
    ),
    ("tensorrt_llm/serve/router.py", "_server_info"): (
        "same-package",
        "serve/router.py Router._server_info",
    ),
    ("tensorrt_llm/serve/disagg_coordinator.py", "_server_info"): (
        "same-package",
        "serve/router.py Router._server_info",
    ),
    ("tensorrt_llm/serve/disagg_coordinator.py", "_local"): (
        "same-package",
        "serve/router.py CoordinatorDelegatingRouter._local",
    ),
    ("tensorrt_llm/serve/openai_disagg_service.py", "_local"): (
        "same-package",
        "serve/router.py CoordinatorDelegatingRouter._local",
    ),
    # _DecodedMediaReservation reads its own module's _DecodedMediaBudget; the
    # generic names are also defined elsewhere, so the index cannot tell.
    ("tensorrt_llm/grpc/openengine/request_mapping.py", "_condition"): (
        "same-package",
        "grpc/openengine/request_mapping.py _DecodedMediaBudget._condition",
    ),
    ("tensorrt_llm/grpc/openengine/request_mapping.py", "_cancelled"): (
        "same-package",
        "grpc/openengine/request_mapping.py _DecodedMediaBudget._cancelled",
    ),
}

FIX_HINT = (
    "Frontends use only public LLM, executor, result and SamplingParams APIs. Add a public "
    "accessor on the class that owns the member (in the owning package) and call it here. "
    "Do not add the access to tests/unittest/serve/private_access_baseline.yaml: the baseline "
    "only shrinks. If the member belongs to this package or to a third-party library, add an "
    "OVERRIDES entry in tests/unittest/serve/test_private_access_lint.py with the reason. "
    "Runtime owners (@NVIDIA/trt-llm-runtime-devs) review both files."
)


def _is_private(name):
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _package_of(rel):
    parts = rel.split("/")
    return parts[1][:-3] if len(parts) == 2 else parts[1]


def _is_self(node):
    if isinstance(node, ast.Name) and node.id in ("self", "cls"):
        return True
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"
    )


class _Candidates(ast.NodeVisitor):
    def __init__(self):
        self.scope, self.rows = [], []

    def _qualname(self):
        return ".".join(self.scope) or "<module>"

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Attribute(self, node):
        if _is_private(node.attr) and not _is_self(node.value):
            self.rows.append((node.lineno, node.attr, ast.unparse(node.value), self._qualname()))
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id in ("getattr", "hasattr", "setattr", "delattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _is_private(node.args[1].value)
            and not _is_self(node.args[0])
        ):
            self.rows.append(
                (node.lineno, node.args[1].value, ast.unparse(node.args[0]), self._qualname())
            )
        self.generic_visit(node)


class _Definitions(ast.NodeVisitor):
    """Record which package defines each private member name."""

    def __init__(self, rel, index):
        self.rel, self.index = rel, index

    def _add(self, name):
        if _is_private(name):
            self.index[name].add(_package_of(self.rel))

    def visit_ClassDef(self, node):
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self._add(target.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self._add(stmt.target.id)
        self.generic_visit(node)

    def _targets(self, targets):
        for target in targets:
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id in ("self", "cls"):
                    self._add(target.attr)
                elif not self.rel.startswith(SCOPES):
                    # `obj_or_class._x = ...` outside the frontends also defines `_x`.
                    self._add(target.attr)
            elif isinstance(target, (ast.Tuple, ast.List)):
                self._targets(target.elts)

    def visit_Assign(self, node):
        self._targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._targets([node.target])
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._targets([node.target])
        self.generic_visit(node)


def _scanned_files(root, prefixes):
    for prefix in prefixes:
        base = root / prefix
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if not rel.startswith(EXCLUDED):
                yield rel


def _name_index(root, names):
    index = collections.defaultdict(set)
    for path in (root / "tensorrt_llm").rglob("*.py*"):
        if path.suffix not in (".py", ".pyi"):
            continue
        text = path.read_text(encoding="utf-8")
        if not any(name in text for name in names):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        _Definitions(path.relative_to(root).as_posix(), index).visit(tree)
    return index


def scan(root=REPO_ROOT, prefixes=SCOPES):
    """Return one row per access: (path, line, member, receiver, scope, verdict, reason)."""
    candidates = []
    for rel in _scanned_files(root, prefixes):
        visitor = _Candidates()
        visitor.visit(ast.parse((root / rel).read_text(encoding="utf-8")))
        candidates.extend((rel,) + row for row in visitor.rows)
    index = _name_index(root, {member for _, _, member, _, _ in candidates})
    rows = []
    for rel, line, member, receiver, scope in candidates:
        package = _package_of(rel)
        if (rel, member) in OVERRIDES:
            verdict, reason = OVERRIDES[(rel, member)]
        else:
            owners = index.get(member, set())
            outside = sorted(owner for owner in owners if owner != package)
            if package in owners and outside:
                verdict, reason = "unresolved", ", ".join(sorted(owners))
            elif package in owners:
                verdict, reason = "same-package", package
            elif outside:
                verdict, reason = "cross", ", ".join(outside)
            else:
                verdict, reason = "third-party", "not defined in tensorrt_llm"
        rows.append((rel, line, member, receiver, scope, verdict, reason))
    return rows


def load_baseline():
    data = yaml.safe_load(BASELINE_FILE.read_text())
    return {(e["path"], e["scope"], e["member"]): e for e in data["entries"]}


def _compare(rows, baseline, prefix=""):
    errors = []
    current = collections.defaultdict(list)
    for row in rows:
        if row[5] == "unresolved":
            errors.append(
                f"Cannot classify `{row[3]}.{row[2]}` at {row[0]}:{row[1]}: `{row[2]}` is defined "
                f"both in this package and elsewhere ({row[6]}). If the receiver is an object from "
                f"another package, this is a new cross-package private access. {FIX_HINT} If it "
                "is this package's own member, add an OVERRIDES entry with verdict same-package."
            )
        elif row[5] == "cross":
            current[(row[0], row[4], row[2])].append(row)
    for key, found in sorted(current.items()):
        allowed = baseline.get(key, {}).get("count", 0)
        if len(found) > allowed:
            where = ", ".join(f"{r[0]}:{r[1]}" for r in found)
            errors.append(
                f"New private access to `{key[2]}` (owned by {found[0][6]}) in {key[1]} "
                f"at {where}: {len(found)} found, baseline allows {allowed}. {FIX_HINT}"
            )
    for key, entry in sorted(baseline.items()):
        if not key[0].startswith(prefix):
            continue
        found = len(current.get(key, []))
        if found < entry["count"]:
            errors.append(
                f"Stale baseline entry: {key[0]} {key[1]} `{key[2]}` lists {entry['count']} "
                f"access(es) but the code has {found}. Lower `count` (delete the entry at 0) in "
                f"{BASELINE_FILE.name} and lower MAX_TOTAL in this PR. If the code moved, change "
                "the entry's path and scope instead; the total must not grow."
            )
    accessed = {(row[0], row[2]) for row in rows}
    for path, member in sorted(OVERRIDES):
        if path.startswith(prefix) and (path, member) not in accessed:
            errors.append(
                f"Stale OVERRIDES entry: {path} `{member}` matches no access. If the code moved, "
                "change the key's path to the new location in this PR, as for a baseline entry; "
                "if the access is gone, delete the entry."
            )
    return errors


def test_no_new_cross_package_private_access():
    errors = _compare(scan(), load_baseline())
    assert not errors, "\n".join(errors)


def test_baseline_only_shrinks():
    total = sum(e["count"] for e in load_baseline().values())
    assert total <= MAX_TOTAL, (
        f"{BASELINE_FILE.name} lists {total} accesses, more than MAX_TOTAL={MAX_TOTAL}. "
        f"The baseline only shrinks. {FIX_HINT}"
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        ("self._x\ncls._x\nsuper()._x()", []),
        ("llm._executor", ["_executor"]),
        ("llm._executor._fatal_error", ["_executor", "_fatal_error"]),
        ("getattr(self._llm, '_check_health', None)", ["_check_health"]),
        ("hasattr(self, '_x')", []),
        ("from m import _private\nobj.__dict__", []),
    ],
)
def test_counting_rule(source, expected):
    visitor = _Candidates()
    visitor.visit(ast.parse(source))
    assert sorted(row[1] for row in visitor.rows) == sorted(expected)


if __name__ == "__main__":
    if "--report" in sys.argv:
        for row in scan():
            print("\t".join(str(x) for x in row))
