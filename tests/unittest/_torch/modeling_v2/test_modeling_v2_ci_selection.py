# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What per-scope CI test selection relies on, read from the source tree.

For a change under ``models/<family>/``, CI runs only the tests of what the change
can reach (``jenkins/scripts/cbts/cbts/rules/modeling_v2.py``): one target, the
targets that import a changed family-level module, or the whole family. A target
is a directory ``models/<family>/<D>/`` that the family's ``routing.py`` returns a
class from. The rule maps paths, not names, so it is safe only while:

* no family reaches another family's code, and no stock code reaches modeling_v2;
* no code reaches into a target except the target itself and the family's
  ``routing.py``;
* test directories mirror source directories: every target has accuracy tests in
  ``accuracy/modeling_v2/<family>/<D>/``, and every accuracy directory names a
  target;
* some test-db entry runs every modeling_v2 unit test, and some test-db or QA list
  entry runs every target's accuracy tests;
* every dynamic import names a module the rule's import analysis can read.

Imports and routing are read with the rule's own parser,
``jenkins/scripts/cbts/cbts/rules/modeling_v2_source.py``, so the two cannot
disagree on what a file imports. This file sits directly under
``tests/unittest/_torch/modeling_v2/`` so CI keeps it for every modeling_v2 change.
It reads files only. Each test takes the repository root as a default argument, so
it can also run against a small fake tree.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

_PY = "tensorrt_llm/_torch/_experimental/modeling_v2"
_CPP = "cpp/tensorrt_llm/modeling_v2"
_UNIT = "tests/unittest/_torch/modeling_v2"
_ACC = "tests/integration/defs/accuracy/modeling_v2"
_TEST_DB = "tests/integration/test_lists/test-db"
_QA_LISTS = "tests/integration/test_lists/qa"
_REPO = Path(__file__).resolve().parents[4]


def _load_source_parser():
    path = _REPO / "jenkins/scripts/cbts/cbts/rules/modeling_v2_source.py"
    spec = importlib.util.spec_from_file_location("_cbts_modeling_v2_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_source = _load_source_parser()
_PACKAGE = _source.PACKAGE
_MODELS_PACKAGE = f"{_PACKAGE}.models"
_NATIVE_SUFFIXES = (".cu", ".cuh", ".h", ".hpp", ".inl", ".cpp")
_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def _families(repo: Path = _REPO) -> list[Path]:
    return sorted(p.parent for p in (repo / _PY / "models").glob("*/__init__.py"))


def _route_returns(family_dir: Path) -> list[tuple[ast.Return, str | None]]:
    """Each return of the family's ``route`` functions, with the target directory it returns."""
    routing = family_dir / "routing.py"
    return _source.route_returns(ast.parse(routing.read_text())) if routing.is_file() else []


def _targets(family_dir: Path) -> list[Path]:
    """The directories the family's ``routing.py`` returns classes from, which makes them targets."""
    names = {target for _, target in _route_returns(family_dir) if target}
    return sorted(family_dir / name for name in names if (family_dir / name).is_dir())


def _package(path: Path, repo: Path) -> str:
    """The package a Python file of the repository sits in."""
    return ".".join(path.parent.relative_to(repo).parts)


def _imports(path: Path, repo: Path) -> list[str]:
    """Every absolute name a Python file imports, ``import_module`` literals included."""
    names, _ = _source.imported_names(ast.parse(path.read_text()), _package(path, repo))
    return sorted(names)


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
    offenders = []
    for path in sorted((repo / "tensorrt_llm").rglob("*.py")):
        if path.is_relative_to(repo / _PY):
            continue
        text = path.read_text(errors="ignore")
        if _STOCK_PY_REFERENCE.search(text) or (
            "modeling_v2" in text
            and any(
                module == _MODELS_PACKAGE or module.startswith(f"{_MODELS_PACKAGE}.")
                for module in _imports(path, repo)
            )
        ):
            offenders.append(str(path.relative_to(repo)))
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
        targets = {p.name for p in _targets(family_dir)}
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
        targets = {p.name for p in _targets(family_dir)}
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
    """``accuracy/modeling_v2/<family>/<D>/`` names a target and holds tests."""
    problems = []
    for family_dir in _families(repo):
        accuracy = repo / _ACC / family_dir.name
        targets = {p.name for p in _targets(family_dir)}
        for d in sorted(accuracy.iterdir()) if accuracy.is_dir() else []:
            if not d.is_dir() or d.name.startswith("_"):
                continue
            where = d.relative_to(repo)
            if d.name not in targets:
                problems.append(
                    f"{where}: {family_dir.relative_to(repo)}/routing.py returns no class "
                    f"from {d.name}/"
                )
            if not any(d.glob("test_*.py")):
                problems.append(f"{where}: no test_*.py")
    assert not problems, "\n".join(problems)


# Families that ship directly constructed components without a registered model.
# An exemption is invalid as soon as routing or registry imports are introduced.
_COMPONENT_ONLY_FAMILIES: set[str] = {"beam"}


def test_every_routed_target_has_accuracy_tests(repo: Path = _REPO):
    """Every class a family's ``route`` returns lives in a target with accuracy tests.

    ``routing.py`` imports each target class with ``from .<D>.<module> import <Class>``,
    so the target directory ``<D>`` is read from that import. CBTS reads targets the
    same way, and tests a family whole while one of its returns is unknown.
    """
    problems = []
    for family_dir in _families(repo):
        routing = family_dir / "routing.py"
        where = routing.relative_to(repo)
        if family_dir.name in _COMPONENT_ONLY_FAMILIES:
            registry = repo / _PY / "models/__init__.py"
            prefix = f"{_MODELS_PACKAGE}.{family_dir.name}"
            imported = any(
                name == prefix or name.startswith(prefix + ".") for name in _imports(registry, repo)
            )
            if routing.exists() or imported:
                problems.append(f"{where}: component-only exemption has routing or registration")
            continue
        if not routing.is_file():
            problems.append(f"{where}: missing")
            continue
        returns = _route_returns(family_dir)
        if not returns:
            problems.append(f"{where}: no return in a @routes(...) function")
        for node, target in returns:
            if target is None:
                problems.append(
                    f"{where}:{node.lineno}: returns something other than a name bound only by "
                    "`from .<D>.<module> import <Class>`, so its target is unknown"
                )
            elif not (repo / _ACC / family_dir.name / target).is_dir():
                problems.append(
                    f"{where}:{node.lineno}: {node.value.id} from {target}/ has no "
                    f"{_ACC}/{family_dir.name}/{target}/"
                )
    assert not problems, "\n".join(problems)


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


def _qa_entries(repo: Path) -> list[str]:
    """The QA list entries: the lines of each text list and the cases of each YAML case list."""
    entries = [
        line.strip()
        for path in sorted((repo / _QA_LISTS).glob("*.txt"))
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for path in sorted((repo / _QA_LISTS).glob("*.yml")):
        data = yaml.safe_load(path.read_text())
        tests = data.get("tests") if isinstance(data, dict) else None
        if isinstance(tests, list):
            entries.extend(
                test["case"]
                for test in tests
                if isinstance(test, dict) and isinstance(test.get("case"), str)
            )
    return entries


# Unit tests that need hardware no CI stage provides, by test-db path, with what they
# wait for. No test-db entry may run them.
_NO_CI_STAGE = {
    "unittest/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/comm": (
        "2 ranks on sm_107; CI has only the single-GPU GR100 stage (auto:gr100-x1)"
    ),
    "unittest/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/comm": (
        "4 ranks on sm_107; CI has only the single-GPU GR100 stage (auto:gr100-x1)"
    ),
    "unittest/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/comm": (
        "2, 4 or 8 ranks on sm_107; CI has only the single-GPU GR100 stage (auto:gr100-x1)"
    ),
}


def test_every_test_is_in_a_test_list(repo: Path = _REPO):
    """Every modeling_v2 test is in a list some job runs.

    A test-db entry runs every unit test file. A test-db or QA list entry runs every
    target's accuracy tests: targets on hardware CI does not have, such as 16-GPU
    deployments, run from the QA lists. Paths are in test-db form: unit tests relative
    to ``tests/``, accuracy tests relative to ``tests/integration/defs/``. Only paths
    are compared: the ``-m cpu_only`` and ``-m "not cpu_only"`` split between stages is
    not checked. Unit tests under ``_NO_CI_STAGE`` are exempt until a stage can run them.
    """
    unit_root, accuracy_root = repo / "tests", repo / "tests/integration/defs"
    test_db = _test_db_entries(repo)
    any_list = test_db + _qa_entries(repo)
    unit_tests = [str(p.relative_to(unit_root)) for p in sorted((repo / _UNIT).rglob("test_*.py"))]
    # An exemption outlives its reason once its tests move, vanish, or a list starts running them.
    stale = sorted(
        where
        for where in _NO_CI_STAGE
        if not any(_at_or_under(t, where) for t in unit_tests)
        or any(_runs(e, t) for e in test_db for t in unit_tests if _at_or_under(t, where))
    )
    assert not stale, f"_NO_CI_STAGE entries with no test or now listed: {stale}"
    # Test-db path -> the test files that stand for it, and the entries that may run them.
    tests = {
        t: ([t], test_db)
        for t in unit_tests
        if not any(_at_or_under(t, where) for where in _NO_CI_STAGE)
    }
    for family_dir in _families(repo):
        for d in _targets(family_dir):
            accuracy = repo / _ACC / family_dir.name / d.name
            if accuracy.is_dir():
                files = [
                    str(p.relative_to(accuracy_root)) for p in sorted(accuracy.rglob("test_*.py"))
                ]
                tests[str(accuracy.relative_to(accuracy_root))] = (files, any_list)
    unreached = [
        path
        for path, (files, entries) in tests.items()
        if not any(_runs(e, f) for e in entries for f in files)
    ]
    assert not unreached, f"no test list runs: {unreached}"


_IMPORTERS = ("import_module", "__import__")
_CATALOG_PACKAGE = f"{_PACKAGE}.catalog"


def _is_str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _starts_with_one_dot(node: ast.AST | None) -> bool:
    """Whether a string literal or f-string starts with exactly one dot."""
    if isinstance(node, ast.JoinedStr):
        node = node.values[0] if node.values else None
    return _is_str(node) and node.value.startswith(".") and not node.value.startswith("..")


def _dynamic_import_problem(call: ast.Call, package: str, loaders: set[str]) -> str | None:
    """Why CBTS cannot read the import ``call`` in ``package`` makes, or None if it can.

    ``loaders`` are the names the file binds to ``importlib.import_module``.
    """
    read = _source.dynamic_import(call, package, loaders)
    if read is None:
        return f"calls {ast.unparse(call.func)}, which CBTS does not read as an import"
    if read:
        if read == _CATALOG_PACKAGE or read.startswith(f"{_CATALOG_PACKAGE}."):
            return "imports the catalog"
        return None
    # CBTS reads a computed name as an import of the caller's own directory.
    func = call.func
    is_import_module = getattr(func, "attr", None) == "import_module" or (
        getattr(func, "id", None) in loaders
    )
    keywords = {k.arg: k.value for k in call.keywords}
    if not is_import_module and (len(call.args) > 4 or "level" in keywords):
        return "passes a level to __import__"
    if not call.args:
        return "passes the module name as a keyword, which CBTS reads as computed"
    anchor = call.args[1] if len(call.args) > 1 else keywords.get("package")
    if not _starts_with_one_dot(call.args[0]):
        return "imports a name that is neither a string literal nor a single-dot relative name"
    if is_import_module and isinstance(anchor, ast.Name) and anchor.id == "__package__":
        return None
    return "imports a relative name without import_module(..., package=__package__)"


# Calls CBTS cannot read but that load only the caller's own target, keyed by
# file and enclosing function. A change inside a target selects the whole target, so
# these hide nothing from CBTS.
_CONFINED_DYNAMIC_IMPORTS = {
    "models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/dispatch.py:get_impl": (
        "loads the target's own kernels package (KERNELS_PACKAGE, modeling._KERNEL_MODULES)"
    ),
}


def test_models_import_dynamically_only_what_cbts_can_read(repo: Path = _REPO):
    """Every ``import_module`` and ``__import__`` call under ``models/`` names a module CBTS can read.

    The CBTS import analysis (``modeling_v2_source.dynamic_import``) reads a
    string-literal name as that module, and reads a computed name as an import of the
    caller's own directory. So a computed name must be a confined relative import: it
    starts with one dot and is resolved with ``package=__package__``. A literal name
    must not name the catalog, which code under ``models/`` imports only with import
    statements. A call the analysis does not read as an import, such as
    ``builtins.__import__``, or a name held in a variable, could load a module whose
    changes then skip the caller's tests.
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
        package = _package(path, repo)
        loaders = _source.import_loaders(tree)
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
                problem = _dynamic_import_problem(call, package, loaders)
                key = f"{path.relative_to(repo / _PY)}:{enclosing.get(id(call), '')}"
                if problem and key in _CONFINED_DYNAMIC_IMPORTS:
                    confined.add(key)
                elif problem:
                    offenders.append(f"{path.relative_to(repo)}:{call.lineno}: {problem}")
    assert not offenders, "\n".join(offenders)
    stale = sorted(set(_CONFINED_DYNAMIC_IMPORTS) - confined)
    assert not stale, f"no longer needed in _CONFINED_DYNAMIC_IMPORTS: {stale}"
