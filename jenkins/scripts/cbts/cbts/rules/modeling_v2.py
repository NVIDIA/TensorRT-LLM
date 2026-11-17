# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ModelingV2Rule — narrows CI when the modeling_v2 subtree changes.

modeling_v2 is a second modeling path living entirely under
`tensorrt_llm/_torch/modeling_v2/`: one self-contained forward per
(checkpoint, GPU arch, parallel topology), assembled from a catalog of
op wrappers. Native code lives beside each model in `csrc/`; shared build
infrastructure lives under `cpp/tensorrt_llm/modeling_v2/` and
`cpp/cmake/modules/modeling_v2.cmake` and selects every modeling_v2 entry.

Block selection: a block belongs to modeling_v2 iff one of its `tests:`
entries matches a marker in `_MV2_ENTRY_PATTERNS`. Those markers are exact
by construction rather than by luck: the unit tests live in their own
`unittest/_torch/modeling_v2/` directory and the accuracy tests in their own
`accuracy/modeling_v2/` directory.

Entry selection: the test tree mirrors the code tree, so each changed file
maps to test paths. The targets are defined by code: a directory
`models/<family>/<target>/` is a target iff a function in the family's
`routing.py` decorated with `@routes(...)` returns a class that `routing.py`
binds with `from .<target>... import <Class>`. Everything else under
`models/<family>/` is family code. If `routing.py` is missing, does not parse,
or has a return whose directory it does not name, every path of the family is
family code, and the whole family's tests stand in for those of its targets.
A target's tests are its unit tests
(`unittest/_torch/modeling_v2/models/<family>/<target>`) and its accuracy
tests (`accuracy/modeling_v2/<family>/<target>`); a family's tests are the
same two paths one level up. The CPU check
`unittest/_torch/modeling_v2/test_modeling_v2_ci_selection.py` requires every
target to have accuracy tests, and every accuracy directory to name a target.

- Any file in a target, native code included, selects only that target's
  tests.
- A Python file of family code in a subdirectory selects its mirror unit
  directory (`unittest/_torch/modeling_v2/models/<family>/<its directory>`).
  When there is none, or the file sits directly in the family directory, it
  selects every family-level unit test instead: the test files directly in
  the family's unit directory and its subdirectories that are not targets.
  Either way it adds the tests of every target that imports it. A file no
  target imports, such as `routing.py`, selects the whole family, as does
  any non-Python family file.
- A Python file under `catalog/<category>/` selects
  `unittest/_torch/modeling_v2/<category>` plus the tests of every target
  that imports it, or every modeling_v2 entry if that directory is missing.
- A Python file of family or catalog code also selects every unit test file
  under `unittest/_torch/modeling_v2/` that imports it, wherever it sits.
- Anything else -- `catalog/_op.py`, the resolver, `models/__init__.py`,
  the shared native build -- can reach every target and selects every
  modeling_v2 entry.

"Imports" is read from the tree on disk, which in CI is the PR head: every
Python file of the subtree and of the unit test directory is parsed once,
with the readings in `modeling_v2_source.py` that the CPU check shares. A
target or test imports a module when one of its files reaches it through
imports anywhere in the files, relative ones, package `__init__` files and
constant `importlib.import_module` calls included, under any name the file
binds `import_module` to. A module depends on its own packages too, since
importing it runs their `__init__` files. A computed `import_module`
argument counts as importing every module under the file's directory. Names
under the subtree count whether or not their file exists, so the importers
of a deleted or moved module are still found. A unit test reaches the
subtree by module name and its helpers (`_validating`, `_cells`, ...) by
bare name; a bare name matches every test-tree file of that name, since tests
put different directories on `sys.path`. A multi-GPU test that runs
`_rank_job.run("<entry>", ...)` reaches `_<entry>_op_matrix.py`, the launcher
its ranks import. Paths stop at the files directly
under the subtree, `models/` or `catalog/`, since those reach every family.
A file that does not parse widens the selection to the whole family or
every entry.

The tests directly under `unittest/_torch/modeling_v2/` check the tree as a
whole (routing, native loading, contracts), so any change keeps them.

A unit test prefix that no entry of a block sits at or under keeps the
longest entry of that block that covers it instead, such as a directory
entry for all of `unittest/_torch/modeling_v2`.

Outward fallback: not needed, and that is a property of the design
rather than an accident. Nothing beyond the resolver module is imported
unless `LLM(modeling_v2=...)` is on: the model loader calls
`modeling_v2_resolve`, which returns immediately when the switch is off,
and the routing modules are imported lazily behind it. The callers
outside the subtree are `tensorrt_llm/_torch/pyexecutor/model_loader.py`
(decides) and `tensorrt_llm/_torch/models/modeling_auto.py` (reads the
decision off `ModelConfig`), neither of which this rule claims -- a PR
touching them falls back to baseline, which is what a change to the
shared resolver deserves.

`.md` exclusion matters more here than for most rules: the catalog
carries a contract document per entry, so roughly a fifth of the files
in the subtree are Markdown. Claiming them would make a
documentation-only PR pull in multi-GPU GB300 stages.

PerfSanity policy: `perfsanity_relevant` is dynamic, True only when a
matched block lives in a `*_perf_sanity*` yaml -- same as VisualGenRule
/ SpecDecRule. modeling_v2 has no perf-sanity entries
today, so this aggregates to False and Groovy Layer 2 drops the
force-keep of `*-PerfSanity-*` stages.

Sanity policy: `sanity_relevant=False`. The subtree ships no
user-facing entry point and is not imported by `trtllm-serve` or by
`import tensorrt_llm`, so nothing it contains is what PackageSanityCheck
verifies about the wheel.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from cbts.blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult
from .docs import is_docs_path
from .modeling_v2_source import PACKAGE, imported_names, package_names, routed_targets

# Source-path prefixes the rule may claim. Tests under tests/** are left
# to TestsDefRule; the two scopes combine via _TESTSONLY_FAMILY.
_MV2_SRC_PREFIXES: tuple[str, ...] = (
    "tensorrt_llm/_torch/modeling_v2/",
    "cpp/tensorrt_llm/modeling_v2/",
)
_CPP_BUILD_FILES = frozenset({"cpp/cmake/modules/modeling_v2.cmake"})
_IGNORE = re.compile(r"--ignore=(\S+)")

_UNIT_ROOT = "unittest/_torch/modeling_v2"
_ACCURACY_ROOT = "accuracy/modeling_v2"
_PKG_MODULE = PACKAGE

# Substrings that mark a test entry as modeling_v2. All are unambiguous:
#   - "unittest/_torch/modeling_v2/" → the op and target unit tests
#   - "accuracy/modeling_v2/" → the per-model accuracy gates
#   - "test_modeling_v2_" → any future unit file named by the subtree's
#     own convention
_MV2_ENTRY_PATTERNS: tuple[str, ...] = (
    f"{_UNIT_ROOT}/",
    f"{_ACCURACY_ROOT}/",
    "test_modeling_v2_",
)


def _is_mv2_claim(path: str) -> bool:
    """Decide whether ModelingV2Rule claims `path`.

    Documentation files are excluded so a contract-only edit does not force
    GPU stages; `DocsRule` routes them to the docs build instead. Other
    suffixes are NOT excluded: a data file under this subtree could be a
    fixture, so the rule keeps claiming it and re-runs the stages (safe
    over-run).
    """
    if not (path.startswith(_MV2_SRC_PREFIXES) or path in _CPP_BUILD_FILES):
        return False
    if is_docs_path(path):
        return False
    return True


def _entry_is_mv2(entry: str) -> bool:
    return entry.split()[0].rstrip("/") == _UNIT_ROOT or any(
        p in entry for p in _MV2_ENTRY_PATTERNS
    )


def _mv2_entries(block) -> list[str]:
    return [t for t in block.tests if _entry_is_mv2(t)]


def _is_whole_tree_test(target: str) -> bool:
    """A test directly under the unit root, which checks the tree as a whole."""
    rest = target.removeprefix(f"{_UNIT_ROOT}/").split("::")[0]
    return target.startswith(f"{_UNIT_ROOT}/") and "/" not in rest and rest.endswith(".py")


def _is_shared(rel: PurePosixPath) -> bool:
    """A file directly under the subtree, `models/` or `catalog/`, which every target can reach."""
    return len(rel.parts) == 1 or (rel.parts[0] in ("models", "catalog") and len(rel.parts) == 2)


def _module_name(rel: PurePosixPath) -> str:
    parts = rel.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((_PKG_MODULE, *parts))


def _parse(path: Path) -> Optional[ast.AST]:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None


def _rank_job_launchers(tree: ast.AST) -> set[str]:
    """The `_<entry>_op_matrix` launchers a test runs with `_rank_job.run("<entry>", ...)`.

    The ranks import the launcher, not the test, so no import names it.
    """
    return {
        f"_{node.args[0].value}_op_matrix"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_rank_job"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


class _ImportGraph:
    """Who imports whom among the subtree's modules and the unit tests, parsed from disk once.

    Unit test files are keyed by their path under the unit root; they import
    the subtree by module name and their helpers by bare name.
    """

    def __init__(
        self, src_root: Path, test_root: Path, parse: Callable[[Path], Optional[ast.AST]]
    ) -> None:
        # Module name -> file path relative to the subtree.
        self.files = {
            _module_name(rel): rel
            for rel in (
                PurePosixPath(p.relative_to(src_root).as_posix())
                for p in sorted(src_root.rglob("*.py"))
            )
        }
        # Path relative to the unit root -> the same path.
        self.tests = {
            rel.as_posix(): rel
            for rel in (
                PurePosixPath(p.relative_to(test_root).as_posix())
                for p in (sorted(test_root.rglob("*.py")) if test_root.is_dir() else ())
            )
        }
        # False when a file does not parse: a path through it would be missed.
        self.complete = True
        self._importers: dict[str, set[str]] = defaultdict(set)
        # Package -> modules that import a computed name, which may be any module under it.
        self._computed: dict[str, set[str]] = defaultdict(set)
        for module, rel in self.files.items():
            tree = parse(src_root / rel)
            if tree is None:
                self.complete = False
                continue
            package = module if rel.name == "__init__.py" else module.rpartition(".")[0]
            names, computed = imported_names(tree, package)
            if computed:
                self._computed[package].add(module)
            # A module depends on its own packages too: importing it runs their __init__.
            for dep in package_names(names | {module}) - {module}:
                self._importers[dep].add(module)
        # A bare name finds every helper of that name: tests put different
        # directories on sys.path, and a wider match only over-selects.
        helpers: dict[str, set[str]] = defaultdict(set)
        for key, rel in self.tests.items():
            helpers[rel.stem].add(key)
        for key, rel in self.tests.items():
            tree = parse(test_root / rel)
            if tree is None:
                self.complete = False
                continue
            # Tests are not a package of the subtree: only absolute names and
            # helper names count, a helper by any part of a dotted name, and a
            # computed name reaches no helper.
            names, _ = imported_names(tree, "")
            deps = package_names(names)
            for name in names | _rank_job_launchers(tree):
                for part in name.split("."):
                    deps |= helpers.get(part, set())
            for dep in deps - {key}:
                self._importers[dep].add(key)

    def _direct_importers(self, name: str) -> set[str]:
        found = set(self._importers.get(name, ()))
        for package, modules in self._computed.items():
            if name == package or name.startswith(f"{package}."):
                found |= modules
        return found

    def importers(self, module: str) -> tuple[set[PurePosixPath], set[PurePosixPath]]:
        """Source files and unit test files that reach `module` through imports.

        Paths do not go through a shared module. Only `test_*.py` files count as
        unit tests; helpers are followed but not returned.
        """
        seen = {module}
        todo = [module]
        while todo:
            for importer in self._direct_importers(todo.pop()):
                if importer in seen or (
                    importer in self.files and _is_shared(self.files[importer])
                ):
                    continue
                seen.add(importer)
                todo.append(importer)
        seen.discard(module)
        sources = {self.files[m] for m in seen if m in self.files}
        tests = {self.tests[k] for k in seen if k in self.tests}
        return sources, {rel for rel in tests if rel.name.startswith("test_")}


def _outermost(prefixes: set[str]) -> set[str]:
    """`prefixes` without the ones under another of them."""
    return {p for p in prefixes if not any(p.startswith(f"{q}/") for q in prefixes)}


def _is_perf_sanity_stem(stem: str) -> bool:
    """True for perf-sanity yaml stems (`l0_*_perf_sanity*`)."""
    return "perf_sanity" in stem


def _kept_entries(entries: list[str], prefixes: set[str]) -> list[str]:
    """The entries of one block that `prefixes` select.

    An entry is kept when it sits at or under a prefix. A prefix with no such
    entry in the block keeps the longest entry that covers it instead.
    """
    targets = {entry: _entry_target(entry).split()[0] for entry in entries}
    kept = {e for e, t in targets.items() if _is_whole_tree_test(t)}
    for prefix in prefixes:
        under = {e for e, t in targets.items() if _target_in_filter_subtree(t, prefix)}
        if not under:
            covering = [
                e
                for e, t in targets.items()
                if _target_in_filter_subtree(prefix, t)
                and not any(
                    _target_in_filter_subtree(prefix, ignored.rstrip("/"))
                    for ignored in _IGNORE.findall(e)
                )
            ]
            under = {max(covering, key=lambda e: len(targets[e]))} if covering else set()
        kept |= under
    return [e for e in entries if e in kept]


class ModelingV2Rule(Rule):
    name = "modelingv2"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage], repo_root: Path) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)
        self._repo_root = Path(repo_root)
        self._src_root = self._repo_root / _MV2_SRC_PREFIXES[0]
        self._graph: Optional[_ImportGraph] = None
        # routing.py trees, which both the targets and the import graph read.
        self._routing_trees: dict[Path, Optional[ast.AST]] = {}
        self._routed: dict[str, Optional[set[str]]] = {}

    def _parse(self, path: Path) -> Optional[ast.AST]:
        if path.name != "routing.py" or path.parent.parent != self._src_root / "models":
            return _parse(path)
        if path not in self._routing_trees:
            self._routing_trees[path] = _parse(path)
        return self._routing_trees[path]

    def _targets(self, family: str) -> Optional[set[str]]:
        """The directories the family's `routing.py` returns target classes from.

        None when `routing.py` is missing, does not parse, or returns something
        whose directory it does not name: every path of the family is then
        family code.
        """
        if family not in self._routed:
            tree = self._parse(self._src_root / "models" / family / "routing.py")
            self._routed[family] = None if tree is None else routed_targets(tree)
        return self._routed[family]

    def _is_target(self, family: str, name: str) -> bool:
        targets = self._targets(family)
        return targets is not None and name in targets

    @staticmethod
    def _target_tests(family: str, target: str) -> set[str]:
        return {f"{_UNIT_ROOT}/models/{family}/{target}", f"{_ACCURACY_ROOT}/{family}/{target}"}

    @staticmethod
    def _family_tests(family: str) -> set[str]:
        return {f"{_UNIT_ROOT}/models/{family}", f"{_ACCURACY_ROOT}/{family}"}

    def _importers(self, rel: PurePosixPath) -> Optional[tuple[set[str], set[str]]]:
        """Test prefixes of every target that imports `rel`, and every unit test file that does.

        An importer in a family whose targets are unknown stands for the whole
        family. None if a file did not parse.
        """
        if self._graph is None:
            self._graph = _ImportGraph(
                self._src_root, self._repo_root / "tests" / _UNIT_ROOT, self._parse
            )
        if not self._graph.complete:
            return None
        sources, tests = self._graph.importers(_module_name(rel))
        targets = set()
        for importer in sources:
            parts = importer.parts
            if parts[0] != "models":
                continue
            if self._targets(parts[1]) is None:
                targets |= self._family_tests(parts[1])
            elif len(parts) > 3 and self._is_target(parts[1], parts[2]):
                targets |= self._target_tests(parts[1], parts[2])
        return targets, {f"{_UNIT_ROOT}/{test}" for test in tests}

    def _family_unit_tests(self, family: str, subdir: PurePosixPath) -> set[str]:
        """The mirror unit directory of family code in `subdir`, else every family-level unit test."""
        unit = f"{_UNIT_ROOT}/models/{family}"
        if subdir.parts and (self._repo_root / "tests" / unit / subdir).is_dir():
            return {f"{unit}/{subdir}"}
        root = self._repo_root / "tests" / unit
        children = sorted(root.iterdir()) if root.is_dir() else []
        return {
            f"{unit}/{child.name}"
            for child in children
            if (child.is_file() and child.name.startswith("test_") and child.suffix == ".py")
            or (
                child.is_dir()
                and child.name != "__pycache__"
                and not self._is_target(family, child.name)
            )
        }

    def _prefixes_for(self, path: str) -> Optional[set[str]]:
        """Filter prefixes a changed file selects; None selects every modeling_v2 entry."""
        if not path.startswith(_MV2_SRC_PREFIXES[0]):
            return None
        rel = PurePosixPath(path[len(_MV2_SRC_PREFIXES[0]) :])
        parts = rel.parts
        if _is_shared(rel):
            return None
        if parts[0] == "catalog":
            unit = f"{_UNIT_ROOT}/{parts[1]}"
            if rel.suffix != ".py" or not (self._repo_root / "tests" / unit).is_dir():
                return None
            importers = self._importers(rel)
            if importers is None:
                return None
            targets, tests = importers
            return _outermost({unit} | targets | tests)
        if parts[0] != "models":
            return None
        family = parts[1]
        if len(parts) > 3 and self._is_target(family, parts[2]):
            return self._target_tests(family, parts[2])
        whole_family = self._family_tests(family)
        if rel.suffix != ".py":
            return whole_family
        importers = self._importers(rel)
        if importers is None:
            return whole_family
        targets, tests = importers
        if not targets or self._targets(family) is None:
            return _outermost(whole_family | tests)
        return _outermost(
            self._family_unit_tests(family, PurePosixPath(*parts[2:-1])) | targets | tests
        )

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if _is_mv2_claim(f)}
        if not claimed:
            return None

        prefixes: Optional[set[str]] = set()
        for path in sorted(claimed):
            selected = self._prefixes_for(path)
            if selected is None:
                prefixes = None
                break
            prefixes |= selected

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            entries = _mv2_entries(block)
            if prefixes is not None:
                entries = _kept_entries(entries, prefixes)
            if not entries:
                continue
            prefix_dict = block_filters.setdefault((block.yaml_stem, block.block_index), {})
            for entry in entries:
                target = _entry_target(entry)
                if target:
                    prefix_dict.setdefault(target, set()).add(entry)

        if not block_filters:
            # Defensive: modeling_v2 source changed but no modeling_v2 block
            # exists in any yaml. Do not fabricate stages -- fall back to
            # baseline so the change still gets coverage. Reachable if the
            # subtree's entries are ever removed from the test-db without
            # the subtree going with them.
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"modelingv2: {len(claimed)} modeling_v2 source file(s); "
                    "no modeling_v2 block matched in any test-db yaml — fallback"
                ),
            )

        affected = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        perfsanity_relevant = any(_is_perf_sanity_stem(stem) for stem, _ in block_filters)

        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="modelingv2only",
            block_filters=block_filters,
            sanity_relevant=False,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"modelingv2: {len(claimed)} modeling_v2 source file(s) → "
                f"{'every modeling_v2 entry' if prefixes is None else f'{len(prefixes)} test prefix(es)'}, "
                f"{len(block_filters)} block(s), {len(affected)} stage(s)"
            ),
        )
