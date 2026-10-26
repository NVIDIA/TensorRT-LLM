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
"""ModelingV2Rule — narrows CI when only modeling_v2 code changes.

modeling_v2 is a second modeling path: one self-contained forward per
(checkpoint, GPU arch, parallel topology), assembled from a catalog of op
wrappers. Its Python lives under `tensorrt_llm/_torch/_experimental/modeling_v2/`
and its native code under `cpp/tensorrt_llm/modeling_v2/`, which builds one
torch-op library per model family, loaded only when routing picks one of the
family's targets.

Scope — per family, else all of modeling_v2:
A change confined to one family's directories (`models/<family>/` in Python,
`cpp/tensorrt_llm/modeling_v2/<familyCamel>/` in C++) selects that family's test
entries -- `accuracy/modeling_v2/test_<family>.py` and
`unittest/_torch/modeling_v2/models/<family>/` -- plus the modeling_v2 CPU
checks. Any other modeling_v2 change (catalog, resolver, loader, the native
build module and its common headers) selects every modeling_v2 entry. Several
families in one PR union their entries.

Why no outward fallback is needed: nothing beyond the resolver is imported, and
no family library is loaded, unless `LLM(modeling_v2=...)` is on -- the model
loader calls the resolver, which returns immediately when the switch is off,
and a target's library opens only when routing imports that target. Family
libraries are separate modules, never linked into libtensorrt_llm.so or
libth_common.so. The claims tests under `tests/unittest/_torch/modeling_v2/`
keep that true (stock never references a target, targets never reach a
sibling, `modeling_v2` defaults to `off`); they run in every scope this rule
selects. The callers outside the subtree (`_torch/pyexecutor/model_loader.py`,
`_torch/models/modeling_auto.py`) stay unclaimed: a change to them runs the
baseline.

`.md` exclusion matters here: every catalog entry carries a contract document,
so a fifth of the Python subtree is Markdown; claiming it would let a
documentation-only PR pull in multi-GPU GB300 stages.

PerfSanity policy: `perfsanity_relevant` is True only when a selected block
lives in a `*_perf_sanity*` yaml -- same as VisualGenRule / SpecDecRule.

Sanity policy: `sanity_relevant=False`. Nothing here is imported by
`trtllm-serve` or `import tensorrt_llm`, and the family libraries are not
loaded by either, so nothing here is what PackageSanityCheck verifies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from cbts.blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult
from .docs import is_docs_path

_PY_ROOT = "tensorrt_llm/_torch/_experimental/modeling_v2/"
_PY_MODELS = _PY_ROOT + "models/"
_CPP_ROOT = "cpp/tensorrt_llm/modeling_v2/"
# Directories of the native tree that are not families.
_CPP_NON_FAMILY_DIRS = frozenset({"common", "catalog"})
# The native build module: shared by every family library.
_CPP_BUILD_FILES = frozenset({"cpp/cmake/modules/modeling_v2.cmake"})

# Source-path prefixes the rule may claim. Tests under tests/** are left to
# TestsDefRule; the two scopes combine via _TESTSONLY_FAMILY.
_MV2_SRC_PREFIXES: tuple[str, ...] = (_PY_ROOT, _CPP_ROOT)

_UNIT_ROOT = "unittest/_torch/modeling_v2"
_ACCURACY_ROOT = "accuracy/modeling_v2"

# Substrings that mark a test entry as modeling_v2. All are unambiguous:
#   - "unittest/_torch/modeling_v2/" → the op-level catalog matrix and the
#     per-family unit tests (`.../models/<family>`)
#   - "accuracy/modeling_v2/" → the per-model accuracy gates
#   - "test_modeling_v2_" → any unit file named by the subtree's own convention
_MV2_ENTRY_PATTERNS: tuple[str, ...] = (
    f"{_UNIT_ROOT}/",
    f"{_ACCURACY_ROOT}/",
    "test_modeling_v2_",
)

# The modeling_v2 CPU checks (claims, routing, native loader): selected in
# every scope, because they are what keeps the narrowing sound.
_MV2_CHECK_ENTRY = re.compile(r"^unittest/_torch/modeling_v2/test_modeling_v2_[\w]+\.py$")


def _camel(name: str) -> str:
    """The C++ directory of a family: `minimax_m3` -> `minimaxM3`.

    Mirrors `_modeling_v2_camel` in cpp/cmake/modules/modeling_v2.cmake.
    """
    parts = [p for p in name.split("_") if p]
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def _snake(camel_name: str) -> str:
    """Best-effort inverse of `_camel`, used only without a repo root."""
    return re.sub(r"(?<=[a-z0-9])([A-Z])", lambda m: "_" + m.group(1).lower(), camel_name)


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
    return not is_docs_path(path)


def _entry_is_mv2(entry: str) -> bool:
    return any(p in entry for p in _MV2_ENTRY_PATTERNS)


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
    targets = {entry: _entry_target(entry) for entry in entries}
    kept = {e for e, t in targets.items() if _is_whole_tree_test(t)}
    for prefix in prefixes:
        under = {e for e, t in targets.items() if _target_in_filter_subtree(t, prefix)}
        if not under:
            covering = [e for e, t in targets.items() if _target_in_filter_subtree(prefix, t)]
            under = {max(covering, key=lambda e: len(targets[e]))} if covering else set()
        kept |= under
        kept |= {e for e, t in targets.items() if _is_ancestor_file(t, prefix)}
    return [e for e in entries if e in kept]


class ModelingV2Rule(Rule):
    name = "modelingv2"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(
        self,
        yaml_index: YAMLIndex,
        stages: dict[str, Stage],
        repo_root: Optional[Path] = None,
    ) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)
        # camelCase native directory -> family, from the Python family packages.
        self._family_of_cpp_dir: dict[str, str] = {}
        if repo_root is not None:
            models = Path(repo_root) / _PY_MODELS
            if models.is_dir():
                for family_dir in models.iterdir():
                    if family_dir.is_dir() and (family_dir / "__init__.py").is_file():
                        self._family_of_cpp_dir[_camel(family_dir.name)] = family_dir.name

    def family_of(self, path: str) -> Optional[str]:
        """The family a claimed path belongs to, or None for shared modeling_v2 code."""
        if path.startswith(_PY_MODELS):
            parts = path[len(_PY_MODELS) :].split("/")
            return parts[0] if len(parts) > 1 else None
        if path.startswith(_CPP_ROOT):
            parts = path[len(_CPP_ROOT) :].split("/")
            if len(parts) < 2 or parts[0] in _CPP_NON_FAMILY_DIRS:
                return None
            return self._family_of_cpp_dir.get(parts[0], _snake(parts[0]))
        return None

    @staticmethod
    def _entry_in_family(target: str, family: str) -> bool:
        accuracy = f"accuracy/modeling_v2/test_{family}.py"
        unit = f"unittest/_torch/modeling_v2/models/{family}"
        return (
            target == accuracy
            or target.startswith(accuracy + "::")
            or target == unit
            or target.startswith(unit + "/")
        )

    def _select(self, families: Optional[set[str]]):
        """Select the modeling_v2 entries a change runs.

        Every modeling_v2 entry when `families` is None, else the entries of
        `families` plus the CPU checks. Also returns the families that matched
        no entry.
        """
        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        matched_families: set[str] = set()
        for block in self.yaml_index.blocks:
            for entry in block.tests:
                if not _entry_is_mv2(entry):
                    continue
                target = _entry_target(entry)
                if not target:
                    continue
                if families is not None:
                    owners = {f for f in families if self._entry_in_family(target, f)}
                    if not owners and not _MV2_CHECK_ENTRY.match(target):
                        continue
                    matched_families |= owners
                key = (block.yaml_stem, block.block_index)
                block_filters.setdefault(key, {}).setdefault(target, set()).add(entry)
        unmatched = set() if families is None else families - matched_families
        return block_filters, unmatched

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if _is_mv2_claim(f)}
        if not claimed:
            return None

        owners = {self.family_of(f) for f in claimed}
        families = None if None in owners else set(owners)
        block_filters, unmatched = self._select(families)

        if not block_filters or unmatched:
            # Do not fabricate stages -- fall back to baseline so the change
            # still gets coverage. Reachable if a family registers no test
            # entry, or the subtree's entries leave the test-db.
            missing = f" for {sorted(unmatched)}" if unmatched else ""
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"modelingv2: {len(claimed)} modeling_v2 source file(s); "
                    f"no modeling_v2 test entry{missing} in any test-db yaml — fallback"
                ),
            )

        affected = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        perfsanity_relevant = any(_is_perf_sanity_stem(stem) for stem, _ in block_filters)
        scope_text = "all modeling_v2" if families is None else f"families {sorted(families)}"

        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="modelingv2only",
            block_filters=block_filters,
            sanity_relevant=False,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"modelingv2: {len(claimed)} modeling_v2 source file(s), {scope_text} → "
                f"{len(block_filters)} modeling_v2 block(s), {len(affected)} stage(s)"
            ),
        )
