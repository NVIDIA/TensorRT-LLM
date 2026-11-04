# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What per-scope CI test selection relies on, read from the source tree.

For a change under ``models/<family>/``, CI runs only the tests of what the change
can reach (``jenkins/scripts/cbts/cbts/rules/modeling_v2.py``): one target, the
targets that import a changed family-level module, or the whole family. A target
is a directory ``models/<family>/<D>/`` with accuracy tests in
``accuracy/modeling_v2/<family>/<D>/``. The rule maps paths, not names, so it is
safe only while:

* no family reaches another family's code, and no stock code reaches modeling_v2;
* no code reaches into a target except the target itself and the family's
  ``routing.py``;
* test directories mirror source directories, and every target a ``route`` can
  return has accuracy tests;
* some test-db entry runs every modeling_v2 test;
* every dynamic import names a module the rule's import analysis can read.

This file sits directly under ``tests/unittest/_torch/modeling_v2/`` so CI keeps it
for every modeling_v2 change. It reads files only. Each test takes the repository
root as a default argument, so it can also run against a small fake tree.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from importlib.util import resolve_name
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

_PY = "tensorrt_llm/_torch/_experimental/modeling_v2"
_CPP = "cpp/tensorrt_llm/modeling_v2"
_UNIT = "tests/unittest/_torch/modeling_v2"
_ACC = "tests/integration/defs/accuracy/modeling_v2"
_TEST_DB = "tests/integration/test_lists/test-db"
_REPO = Path(__file__).resolve().parents[4]
_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"
_MODELS_PACKAGE = f"{_PACKAGE}.models"
_NATIVE_SUFFIXES = (".cu", ".cuh", ".h", ".hpp", ".inl", ".cpp")
_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def _families(repo: Path = _REPO) -> list[Path]:
    return sorted(p.parent for p in (repo / _PY / "models").glob("*/__init__.py"))


def _targets(family_dir: Path, repo: Path = _REPO) -> list[Path]:
    """The family's directories that have accuracy tests, which is what makes them targets."""
    accuracy = repo / _ACC / family_dir.name
    return sorted(p for p in family_dir.iterdir() if p.is_dir() and (accuracy / p.name).is_dir())


def _imported_modules(node: ast.AST, package: str) -> list[str]:
    """The absolute names an import statement in ``package`` reaches, one per imported name."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level:
            module = resolve_name("." * node.level + module, package)
        return [f"{module}.{alias.name}" for alias in node.names]
    return []


def _imports(path: Path, repo: Path) -> list[str]:
    """Every absolute name a Python file of the package imports."""
    package = ".".join((_PACKAGE, *path.parent.relative_to(repo / _PY).parts))
    return [
        module
        for node in ast.walk(ast.parse(path.read_text()))
        for module in _imported_modules(node, package)
    ]


def test_families_do_not_reach_each_other(repo: Path = _REPO):
    """A family's Python imports and C++ includes stay inside the family."""
    families = {p.name for p in _families(repo)}
    reaches = []
    for family_dir in _families(repo):
        family = family_dir.name
        for path in sorted(family_dir.rglob("*")):
            where = path.relative_to(repo)
            if path.suffix == ".py":
                for module in _imports(path, repo):
                    if module.startswith(_MODELS_PACKAGE + "."):
                        owner = module[len(_MODELS_PACKAGE) + 1 :].split(".")[0]
                        if owner != family:
                            reaches.append(f"{where} imports {module}")
            elif path.suffix in _NATIVE_SUFFIXES:
                for include in _INCLUDE.findall(path.read_text(errors="ignore")):
                    owner = include.split("/", 1)[0]
                    if owner in families and owner != family:
                        reaches.append(f"{where} includes {include}")
    assert not reaches, "\n".join(reaches)


_STOCK_PY_REFERENCE = re.compile(
    r"modeling_v2[./]models\b|torch\.ops\.mv2_|register_fake\(\s*f?[\"']mv2_"
)
_STOCK_CPP_INCLUDE = re.compile(r'#\s*include\s*"tensorrt_llm/modeling_v2/')


def test_stock_code_does_not_reach_modeling_v2(repo: Path = _REPO):
    """Stock code reaches modeling_v2 only through the resolver.

    No family import, op call or fake registration in Python, and no modeling_v2
    header in C++.
    """
    offenders = [
        str(path.relative_to(repo))
        for path in sorted((repo / "tensorrt_llm").rglob("*.py"))
        if not path.is_relative_to(repo / _PY)
        and _STOCK_PY_REFERENCE.search(path.read_text(errors="ignore"))
    ]
    for root in ("cpp/tensorrt_llm", "cpp/include", "cpp/tests"):
        for path in sorted((repo / root).rglob("*")):
            # Generated cubin sources are large and include nothing of interest.
            if (
                path.is_file()
                and path.suffix in _NATIVE_SUFFIXES
                and not path.is_relative_to(repo / _CPP)
                and path.stat().st_size < (1 << 20)
                and _STOCK_CPP_INCLUDE.search(path.read_text(errors="ignore"))
            ):
                offenders.append(str(path.relative_to(repo)))
    assert not offenders, f"stock code reaches modeling_v2: {offenders}"


def test_only_routing_reaches_into_targets(repo: Path = _REPO):
    """Only a target itself and its family's ``routing.py`` import or include the target's code.

    A change inside a target selects only that target's tests. A sibling target that
    reached into it, directly or through family-level code it imports, would go
    untested.
    """
    reaches = []
    for family_dir in _families(repo):
        family = family_dir.name
        package = f"{_MODELS_PACKAGE}.{family}."
        targets = {p.name for p in _targets(family_dir, repo)}
        for path in sorted(family_dir.rglob("*")):
            relative = path.relative_to(family_dir)
            if relative == Path("routing.py"):
                continue
            # Inside a target: the other targets. At family level: every target.
            others = targets - {relative.parts[0]}
            where = path.relative_to(repo)
            if path.suffix == ".py":
                for module in _imports(path, repo):
                    if (
                        module.startswith(package)
                        and module[len(package) :].split(".")[0] in others
                    ):
                        reaches.append(f"{where} imports {module}")
            elif path.suffix in _NATIVE_SUFFIXES:
                for include in _INCLUDE.findall(path.read_text(errors="ignore")):
                    if include.startswith(f"{family}/") and include.split("/")[1] in others:
                        reaches.append(f"{where} includes {include}")
    assert not reaches, "\n".join(reaches)


def test_test_directories_name_a_family(repo: Path = _REPO):
    """Every directory of unit tests under ``models/`` and of accuracy tests names a family."""
    families = {p.name for p in _families(repo)}
    stray = [
        str(d.relative_to(repo))
        for root in (repo / _UNIT / "models", repo / _ACC)
        for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith("_") and d.name not in families
    ]
    assert not stray, f"test directories named after no modeling_v2 family: {stray}"


def test_unit_test_directories_mirror_source(repo: Path = _REPO):
    """``unittest/.../models/<family>/<path>/`` holds the tests of ``models/<family>/<path>/``.

    CBTS maps a changed family-level source directory to the unit test directory at
    the same path. A change inside a target selects the whole target, so the layout
    inside a target's test directory is free. ``_``-prefixed directories, such as
    ``__pycache__``, are skipped.
    """
    unmatched = []
    for family_dir in _families(repo):
        tests = repo / _UNIT / "models" / family_dir.name
        targets = {p.name for p in _targets(family_dir, repo)}
        for d in sorted(tests.rglob("*")):
            relative = d.relative_to(tests)
            if (
                d.is_dir()
                and not any(part.startswith("_") for part in relative.parts)
                and relative.parts[0] not in targets
                and not (family_dir / relative).is_dir()
            ):
                unmatched.append(str(d.relative_to(repo)))
    assert not unmatched, f"unit test directories with no source directory: {unmatched}"


def test_accuracy_directories_are_targets(repo: Path = _REPO):
    """``accuracy/modeling_v2/<family>/<D>/`` names a source directory and holds tests."""
    problems = []
    for family_dir in _families(repo):
        accuracy = repo / _ACC / family_dir.name
        for d in sorted(accuracy.iterdir()) if accuracy.is_dir() else []:
            if not d.is_dir() or d.name.startswith("_"):
                continue
            where = d.relative_to(repo)
            if not (family_dir / d.name).is_dir():
                problems.append(f"{where}: no {family_dir.relative_to(repo)}/{d.name}/")
            if not any(d.glob("test_*.py")):
                problems.append(f"{where}: no test_*.py")
    assert not problems, "\n".join(problems)


def _route_returns(routing: Path) -> list[ast.Return]:
    """The ``return`` statements of the functions in ``routing`` decorated with ``@routes(...)``."""
    returns = []
    for function in ast.walk(ast.parse(routing.read_text())):
        if isinstance(function, ast.FunctionDef) and any(
            isinstance(d, ast.Call)
            and getattr(d.func, "id", getattr(d.func, "attr", None)) == "routes"
            for d in function.decorator_list
        ):
            # Only the function's own returns, not those of a function nested in it.
            todo = list(function.body)
            while todo:
                node = todo.pop()
                if isinstance(node, ast.Return):
                    returns.append(node)
                elif not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
                ):
                    todo.extend(ast.iter_child_nodes(node))
    return sorted(returns, key=lambda node: node.lineno)


def test_every_routed_target_has_accuracy_tests(repo: Path = _REPO):
    """Every class a family's ``route`` returns lives in a directory with accuracy tests.

    ``routing.py`` imports each target class with ``from .<D>.<module> import <Class>``,
    so the target directory ``<D>`` is read from that import.
    """
    problems = []
    for family_dir in _families(repo):
        routing = family_dir / "routing.py"
        where = routing.relative_to(repo)
        if not routing.is_file():
            problems.append(f"{where}: missing")
            continue
        # Name -> the first package of the ``from .<package>... import <name>`` that binds it.
        bound = {
            alias.asname or alias.name: node.module.split(".")[0]
            for node in ast.walk(ast.parse(routing.read_text()))
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
            for alias in node.names
        }
        returns = _route_returns(routing)
        if not returns:
            problems.append(f"{where}: no return in a @routes(...) function")
        for node in returns:
            name = node.value.id if isinstance(node.value, ast.Name) else None
            if name not in bound:
                problems.append(
                    f"{where}:{node.lineno}: returns something other than a name bound by "
                    "`from .<D>.<module> import <Class>`, so its target is unknown"
                )
            elif not (repo / _ACC / family_dir.name / bound[name]).is_dir():
                problems.append(
                    f"{where}:{node.lineno}: {name} from {bound[name]}/ has no "
                    f"{_ACC}/{family_dir.name}/{bound[name]}/"
                )
    assert not problems, "\n".join(problems)


# Test-db paths that no test-db entry runs, and why. An entry that becomes reached,
# or names a path that no longer exists, fails the test, so the list stays current.
_UNREACHED = {
    "accuracy/modeling_v2/glm_moe_dsa/glm_5_2_nvfp4__sm_107__tp16": (
        "no test-db entry yet (SM 10.7, TP16); run manually"
    ),
    "accuracy/modeling_v2/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4": (
        "only in qa/llm_function_multinode.txt, not in CI"
    ),
    "accuracy/modeling_v2/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1": (
        "only in qa/llm_function_multinode.txt, not in CI"
    ),
}


def _test_db_entries(repo: Path) -> list[str]:
    entries = []
    for path in sorted((repo / _TEST_DB).glob("*.yml")):
        for blocks in (yaml.safe_load(path.read_text()) or {}).values():
            if isinstance(blocks, list):
                entries.extend(test for block in blocks for test in block.get("tests") or [])
    return entries


def _at_or_under(path: str, where: str) -> bool:
    where = where.rstrip("/")
    return path == where or path.startswith(where + "/")


def _runs(entry: str, path: str) -> bool:
    """Whether a test-db entry runs the test file ``path``, both in test-db form."""
    where, *options = entry.split()
    if not _at_or_under(path, where.split("::")[0]):
        return False
    for token in options:
        if token.startswith("--ignore=") and _at_or_under(path, token[len("--ignore=") :]):
            return False
        # pytest matches --ignore-glob with fnmatch, where ``*`` also matches ``/``.
        elif token.startswith("--ignore-glob=") and fnmatch.fnmatch(
            path, token[len("--ignore-glob=") :]
        ):
            return False
    return True


def test_every_test_is_in_test_db(repo: Path = _REPO):
    """Some test-db entry runs every unit test file and every target's accuracy tests.

    Paths are in test-db form: unit tests relative to ``tests/``, accuracy tests
    relative to ``tests/integration/defs/``. Only paths are compared: the
    ``-m cpu_only`` and ``-m "not cpu_only"`` split between stages is not checked.
    """
    unit_root, accuracy_root = repo / "tests", repo / "tests/integration/defs"
    # Test-db path -> the test files that stand for it.
    tests = {
        str(p.relative_to(unit_root)): [str(p.relative_to(unit_root))]
        for p in sorted((repo / _UNIT).rglob("test_*.py"))
    }
    for family_dir in _families(repo):
        for d in _targets(family_dir, repo):
            accuracy = repo / _ACC / family_dir.name / d.name
            tests[str(accuracy.relative_to(accuracy_root))] = [
                str(p.relative_to(accuracy_root)) for p in sorted(accuracy.rglob("test_*.py"))
            ]
    entries = _test_db_entries(repo)
    run = {path: any(_runs(e, f) for e in entries for f in files) for path, files in tests.items()}
    unreached = [path for path, hit in run.items() if not hit and path not in _UNREACHED]
    stale = [path for path in _UNREACHED if run.get(path, True)]
    assert not unreached, f"no test-db entry runs: {unreached}"
    assert not stale, f"_UNREACHED entries that are run or no longer exist: {stale}"


_IMPORTERS = ("import_module", "__import__")


def _is_str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _starts_with_one_dot(node: ast.AST | None) -> bool:
    """Whether a string literal or f-string starts with exactly one dot."""
    if isinstance(node, ast.JoinedStr):
        node = node.values[0] if node.values else None
    return _is_str(node) and node.value.startswith(".") and not node.value.startswith("..")


def _dynamic_import_problem(call: ast.Call, aliases: set[str]) -> str | None:
    """Why CBTS cannot read the import ``call`` makes, or None if it can.

    ``aliases`` are the other names the file binds to an importer function.
    """
    func = call.func
    if isinstance(func, ast.Name) and func.id in aliases:
        return f"calls {func.id}, an alias CBTS does not read"
    if isinstance(func, ast.Attribute) and func.attr == "__import__":
        return "calls __import__ as an attribute, which CBTS does not read"
    is_import_module = getattr(func, "id", getattr(func, "attr", None)) == "import_module"
    keywords = {k.arg: k.value for k in call.keywords}
    name = call.args[0] if call.args else keywords.get("name")
    package = call.args[1] if len(call.args) > 1 else keywords.get("package")
    if _starts_with_one_dot(name):
        if is_import_module and isinstance(package, ast.Name) and package.id == "__package__":
            return None
        return "imports a relative name without import_module(..., package=__package__)"
    if not call.args:
        return "passes the module name as a keyword, which CBTS reads as computed"
    if not _is_str(name) or name.value.startswith("."):
        return "imports a name that is neither a string literal nor a single-dot relative name"
    if "catalog" in name.value:
        return "imports the catalog"
    if not is_import_module and (len(call.args) > 4 or "level" in keywords):
        return "passes a level to __import__"
    return None


# Calls the rule below cannot read but that load only the caller's own target, keyed by
# file and enclosing function. A change inside a target selects the whole target, so
# these hide nothing from CBTS.
_CONFINED_DYNAMIC_IMPORTS = {
    "models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/dispatch.py:get_impl": (
        "loads the target's own kernels package (KERNELS_PACKAGE, modeling._KERNEL_MODULES)"
    ),
}


def test_models_import_dynamically_only_what_cbts_can_read(repo: Path = _REPO):
    """Every ``import_module`` and ``__import__`` call under ``models/`` names a module CBTS can read.

    The CBTS import analysis reads a string-literal name as that module, and reads a
    computed name as an import of the caller's own directory. So a computed name must
    be a confined relative import: it starts with one dot and is resolved with
    ``package=__package__``. A literal name must not name the catalog, which code
    under ``models/`` imports only with import statements. A call through an alias,
    or with a name held in a variable, could load a module whose changes then skip
    the caller's tests.
    """
    offenders = []
    confined = set()
    for path in sorted((repo / _PY / "models").rglob("*.py")):
        tree = ast.parse(path.read_text())
        enclosing = {
            id(node): func.name
            for func in ast.walk(tree)
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(func)
        }
        # ``from importlib import import_module as x``: ``x`` calls an importer.
        aliases = {
            alias.asname
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module in ("importlib", "builtins")
            for alias in node.names
            if alias.name in _IMPORTERS and alias.asname not in (None, alias.name)
        }
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and (
                getattr(call.func, "id", None) in (*_IMPORTERS, *aliases)
                or getattr(call.func, "attr", None) in _IMPORTERS
            ):
                problem = _dynamic_import_problem(call, aliases)
                key = f"{path.relative_to(repo / _PY)}:{enclosing.get(id(call), '')}"
                if problem and key in _CONFINED_DYNAMIC_IMPORTS:
                    confined.add(key)
                elif problem:
                    offenders.append(f"{path.relative_to(repo)}:{call.lineno}: {problem}")
    assert not offenders, "\n".join(offenders)
    stale = sorted(set(_CONFINED_DYNAMIC_IMPORTS) - confined)
    assert not stale, f"no longer needed in _CONFINED_DYNAMIC_IMPORTS: {stale}"
