# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""AutoDeployRule — narrows CI when AutoDeploy source paths change.

AutoDeploy is a beta backend with isolated CI stages (`*-AutoDeploy-*`).
The non-AD code that references AD does so via lazy imports guarded by
`if backend == "_autodeploy"`, so AD-only source changes don't affect
the main PyTorch backend's tests. `scripts/check_auto_deploy_imports.py`
enforces AD's outbound import discipline.

Block selection (see RULES_BACKLOG.md P1 Output):
- Primary: `condition.terms.backend == 'autodeploy'`.
- Supplementary: blocks containing entries with `test_llm_api_autodeploy.py`
  in the path or `_autodeploy-` in the parametrize id. These cover the
  3 entries that live in `backend: pytorch` blocks because Jenkins has
  no `L40S-AutoDeploy-*` / `H100-Perf-AutoDeploy-*` stage.
"""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex, _entry_target

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult

# Source paths AutoDeployRule claims. Tests under tests/unittest/auto_deploy/
# and tests/integration/defs/accuracy/test_llm_api_autodeploy.py are left to
# TestsDefRule; the two rules' scopes combine via _TESTSONLY_FAMILY.
_AD_SRC_PREFIXES: tuple[str, ...] = (
    "examples/auto_deploy/",
    "tensorrt_llm/_torch/auto_deploy/",
)

# Stable substrings marking a test entry as AD-only even when the
# enclosing block's condition is not `backend: autodeploy`.
# - `test_llm_api_autodeploy.py`: AD accuracy test filename convention.
# - `_autodeploy-`: cross-codebase parametrize id used by trtllm-bench
#   and the literal trigger string for AD's lazy imports in
#   `commands/serve.py`.
# Audit (2026-05): exactly 3 entries in 2 blocks rely on this match.
_AD_LEAKER_PATTERNS: tuple[str, ...] = (
    "test_llm_api_autodeploy.py",
    "_autodeploy-",
)


def _is_ad_claim(path: str) -> bool:
    """Decide whether AutoDeployRule claims `path`.

    `*.md` files are excluded so docs-only PRs (e.g.
    `examples/auto_deploy/README.md`) don't force AD stages —
    `OutOfScopeRule` claims them as noop instead. Other suffixes
    (`.png` / `.jpg` / etc.) are NOT excluded here: a binary asset
    under an AD path could be a test fixture, so the rule keeps
    claiming them and forces AD stages to re-run (safe over-run).
    """
    if not path.startswith(_AD_SRC_PREFIXES):
        return False
    if path.endswith(".md"):
        return False
    return True


def _block_backend(block) -> Optional[str]:
    cond = block.condition if isinstance(block.condition, dict) else {}
    terms = cond.get("terms", {})
    return terms.get("backend") if isinstance(terms, dict) else None


def _entry_is_ad_leaker(entry: str) -> bool:
    return any(p in entry for p in _AD_LEAKER_PATTERNS)


def _ad_entries(block) -> list[str]:
    """Entries in `block` to keep when the block is matched as AD.

    For `backend: autodeploy` blocks: every entry (block is AD-pure).
    For leaker blocks (e.g. `backend: pytorch` containing AD test
    entries): only the entries that match the leaker patterns.
    Non-AD siblings stay governed by other rules.
    """
    if _block_backend(block) == "autodeploy":
        return list(block.tests)
    return [t for t in block.tests if _entry_is_ad_leaker(t)]


def _is_perf_stem(stem: str) -> bool:
    return stem == "l0_perf" or "perf_sanity" in stem


class AutoDeployRule(Rule):
    name = "autodeploy"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if _is_ad_claim(f)}
        if not claimed:
            return None

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            entries = _ad_entries(block)
            if not entries:
                continue
            key = (block.yaml_stem, block.block_index)
            prefix_dict = block_filters.setdefault(key, {})
            for entry in entries:
                target = _entry_target(entry)
                if target:
                    prefix_dict.setdefault(target, set()).add(entry)

        if not block_filters:
            # Defensive: AD source changed but no AD blocks exist in any
            # yaml. Don't fabricate stages — fall back to baseline so the
            # change still gets coverage.
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"autodeploy: {len(claimed)} AD source file(s); "
                    "no AD block matched in any test-db yaml — fallback"
                ),
            )

        affected = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        perfsanity_relevant = any(_is_perf_stem(stem) for stem, _ in block_filters)

        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="autodeployonly",
            block_filters=block_filters,
            sanity_relevant=False,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"autodeploy: {len(claimed)} AD source file(s) → "
                f"{len(block_filters)} AD block(s), {len(affected)} stage(s)"
            ),
        )
