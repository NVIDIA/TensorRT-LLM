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
op wrappers.

Block selection — entry-pattern based only:
It has no `condition.terms.backend` of its own; its entries sit in
`backend: pytorch` blocks beside everything else. A block belongs to
modeling_v2 iff one of its `tests:` entries matches a marker in
`_MV2_ENTRY_PATTERNS`. Those markers are exact by construction rather
than by luck: every unit test file in the subtree is named
`test_modeling_v2_*` precisely so it cannot collide with the upstream
test of the same op, and the accuracy files follow the same prefix. So
there is no substring that could claim an unrelated entry, and no
`mtp_nextn=0`-style carve-out is needed.

Outward fallback: not needed, and that is a property of the design
rather than an accident. Nothing imports this subtree unless
`TRTLLM_MODELING_V2` is set: `AutoModelForCausalLM._resolve_class` calls
`modeling_v2_resolve`, which returns immediately when the switch is off,
and the routing modules are imported lazily behind it. The one caller
outside the subtree is `tensorrt_llm/_torch/models/modeling_auto.py`,
which this rule does not claim -- a PR touching it falls back to
baseline, which is what a change to the shared resolver deserves.

`.md` exclusion matters more here than for most rules: the catalog
carries a contract document per entry, so roughly a fifth of the files
in the subtree are Markdown. Claiming them would make a
documentation-only PR pull in multi-GPU GB300 stages.

PerfSanity policy: `perfsanity_relevant` is dynamic, True only when a
matched block lives in a `*_perf_sanity*` yaml -- same as AutoDeployRule
/ VisualGenRule / SpecDecRule. modeling_v2 has no perf-sanity entries
today, so this aggregates to False and Groovy Layer 2 drops the
force-keep of `*-PerfSanity-*` stages.

Sanity policy: `sanity_relevant=False`. The subtree ships no
user-facing entry point and is not imported by `trtllm-serve` or by
`import tensorrt_llm`, so nothing it contains is what PackageSanityCheck
verifies about the wheel.
"""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex, _entry_target

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult

# Source-path prefixes the rule may claim. Tests under tests/** are left
# to TestsDefRule; the two scopes combine via _TESTSONLY_FAMILY.
_MV2_SRC_PREFIXES: tuple[str, ...] = ("tensorrt_llm/_torch/modeling_v2/",)

# Substrings that mark a test entry as modeling_v2. Both are unambiguous:
#   - "unittest/_torch/modeling_v2/" → the op-level catalog matrix, taken
#     as whole-directory entries (one on l0_gb300, one on
#     l0_gb300_multi_gpus for the 4-rank collectives)
#   - "test_modeling_v2_" → the accuracy gates, and any future unit file
#     named by the subtree's own convention
_MV2_ENTRY_PATTERNS: tuple[str, ...] = (
    "unittest/_torch/modeling_v2/",
    "test_modeling_v2_",
)


def _is_mv2_claim(path: str) -> bool:
    """Decide whether ModelingV2Rule claims `path`.

    `*.md` is excluded so a contract-only edit does not force GPU stages
    -- `OutOfScopeRule` claims those as noop instead. Other suffixes are
    NOT excluded: a data file under this subtree could be a fixture, so
    the rule keeps claiming it and re-runs the stages (safe over-run).
    """
    if not path.startswith(_MV2_SRC_PREFIXES):
        return False
    if path.endswith(".md"):
        return False
    return True


def _entry_is_mv2(entry: str) -> bool:
    return any(p in entry for p in _MV2_ENTRY_PATTERNS)


def _mv2_entries(block) -> list[str]:
    return [t for t in block.tests if _entry_is_mv2(t)]


def _is_perf_sanity_stem(stem: str) -> bool:
    """True for perf-sanity yaml stems (`l0_*_perf_sanity*`)."""
    return "perf_sanity" in stem


class ModelingV2Rule(Rule):
    name = "modelingv2"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if _is_mv2_claim(f)}
        if not claimed:
            return None

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            entries = _mv2_entries(block)
            if not entries:
                continue
            key = (block.yaml_stem, block.block_index)
            prefix_dict = block_filters.setdefault(key, {})
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
                f"{len(block_filters)} modeling_v2 block(s), {len(affected)} stage(s)"
            ),
        )
