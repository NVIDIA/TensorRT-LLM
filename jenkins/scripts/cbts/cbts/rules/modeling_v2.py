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
`tensorrt_llm/_torch/_experimental/modeling_v2/`: one self-contained forward per
(checkpoint, GPU arch, parallel topology), assembled from a catalog of
op wrappers. Native code lives beside each model in `csrc/`; shared build
infrastructure lives under `cpp/tensorrt_llm/modeling_v2/` and
`cpp/cmake/modules/modeling_v2.cmake` and selects every modeling_v2 entry.

Block selection: a block belongs to modeling_v2 iff one of its `tests:`
entries matches a marker in `_MV2_ENTRY_PATTERNS`. Those markers are exact
by construction rather than by luck: every unit test file in the subtree is
named `test_modeling_v2_*` precisely so it cannot collide with the upstream
test of the same op, and the accuracy gates live in their own
`accuracy/modeling_v2/` directory.

Entry selection: each changed file maps to the tests it can affect.

- `models/` is a tree of nodes. A directory is a node when it, or a
  directory below it, holds a target's `modeling.py`; a family directory
  (`models/<family>/`) is always one. Other directories -- `ops/` and the
  like -- serve the node above them. A changed file belongs to the nearest
  node at or above it, and selects that node's unit tests
  (`unittest/_torch/modeling_v2/models/<node>`) plus the accuracy gate of
  every target under it. A target's gate is `Test<TargetClass>` in
  `accuracy/modeling_v2/test_<family>.py`, the class name read from the
  import line in the family's `routing.py` that routes to it.
- Anything else -- the catalog, the resolver, `models/__init__.py` -- can
  reach every target and selects every modeling_v2 entry.

The tests directly under `unittest/_torch/modeling_v2/` check the tree as a
whole (claims, routing, contracts), so any change keeps them.

A unit test prefix that no entry of a block sits at or under keeps the
longest entry of that block that covers it instead: a family is one
directory entry, and a change to one of its targets runs the family's
unit tests. A test file listed at a directory above a node -- a family's
own test file -- belongs to every node under that directory, so a change
to any of them keeps it. Whenever the tree on disk cannot answer -- a node directory
deleted with its `routing.py` -- the file falls back to the wider
selection. So does a node holding a `modeling.py` that its family's
`routing.py` does not import, such as a target nested below the family's
first level: its accuracy gate cannot be named.

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

import re
from pathlib import Path
from typing import Optional

from cbts.blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult
from .docs import is_docs_path

# Source-path prefixes the rule may claim. Tests under tests/** are left
# to TestsDefRule; the two scopes combine via _TESTSONLY_FAMILY.
_MV2_SRC_PREFIXES: tuple[str, ...] = (
    "tensorrt_llm/_torch/_experimental/modeling_v2/",
    "cpp/tensorrt_llm/modeling_v2/",
)
_CPP_BUILD_FILES = frozenset({"cpp/cmake/modules/modeling_v2.cmake"})
_IGNORE = re.compile(r"--ignore=(\S+)")

_UNIT_ROOT = "unittest/_torch/modeling_v2"
_ACCURACY_ROOT = "accuracy/modeling_v2"

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

# A family's `routing.py` imports each target it routes to by this line, from
# the target directory directly under the family. The claims test reads the
# same line with the same pattern.
_TARGET_IMPORT = re.compile(
    r"^\s*from \.(\w+)\.modeling import (?:(\w+)|\(\s*(\w+),?\s*\))$", re.MULTILINE
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


def _is_ancestor_file(target: str, prefix: str) -> bool:
    """A test file in a directory below the models root that holds `prefix`.

    A test file at a family (or an intermediate) directory belongs to every
    node under it, so a change to any of those nodes runs it.
    """
    parent, _, name = target.split("::")[0].rpartition("/")
    return (
        name.endswith(".py")
        and parent.startswith(f"{_UNIT_ROOT}/models/")
        and _target_in_filter_subtree(prefix, parent)
    )


def _is_perf_sanity_stem(stem: str) -> bool:
    """True for perf-sanity yaml stems (`l0_*_perf_sanity*`)."""
    return "perf_sanity" in stem


def _kept_entries(entries: list[str], prefixes: set[str]) -> list[str]:
    """The entries of one block that `prefixes` select.

    An entry is kept when it sits at or under a prefix. A prefix with no such
    entry in the block keeps the longest entry that covers it instead. A test
    file in a directory above a prefix is kept as well; it does not count as
    sitting under the prefix, so it never displaces the covering entry.
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
        kept |= {e for e, t in targets.items() if _is_ancestor_file(t, prefix)}
    return [e for e in entries if e in kept]


class ModelingV2Rule(Rule):
    name = "modelingv2"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage], repo_root: Path) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)
        self._src_root = Path(repo_root) / _MV2_SRC_PREFIXES[0]
        self._tests_root = Path(repo_root) / "tests"

    def _is_node(self, rel_dir: Path) -> bool:
        """Whether `models/<rel_dir>` is a node: a family, or holds a target at or below it."""
        path = self._src_root / "models" / rel_dir
        return len(rel_dir.parts) == 1 or (path.is_dir() and any(path.rglob("modeling.py")))

    def _gates_under(self, node: Path) -> Optional[set[str]]:
        """Accuracy-gate prefixes of every target at or under `node`.

        None if unreadable: no `routing.py`, or a `modeling.py` under `node`
        that no import line in it routes to, whose gate cannot be named.
        """
        models = self._src_root / "models"
        family = node.parts[0]
        routing = models / family / "routing.py"
        if not routing.is_file():
            return None
        routed, gates = set(), set()
        for target, plain, wrapped in _TARGET_IMPORT.findall(routing.read_text(encoding="utf-8")):
            target_cls = plain or wrapped
            leaf = Path(family, target)
            routed.add(leaf)
            if leaf == node or node in leaf.parents:
                gates.add(f"{_ACCURACY_ROOT}/test_{family}.py::Test{target_cls}")
        if any(
            m.parent.relative_to(models) not in routed for m in (models / node).rglob("modeling.py")
        ):
            return None
        return gates

    def _prefixes_for(self, path: str) -> Optional[set[str]]:
        """Filter prefixes a changed file selects; None selects every modeling_v2 entry."""
        if not path.startswith(_MV2_SRC_PREFIXES[0]):
            return None
        rel = Path(path[len(_MV2_SRC_PREFIXES[0]) :])
        if rel.parts[0] != "models" or len(rel.parts) < 3:
            return None
        node = Path(*rel.parts[1:-1])
        while not self._is_node(node):
            node = node.parent
        gates = self._gates_under(node)
        if gates is None:
            return None
        return {f"{_UNIT_ROOT}/models/{node.as_posix()}"} | gates

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
