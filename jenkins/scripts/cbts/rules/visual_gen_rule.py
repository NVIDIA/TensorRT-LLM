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
"""VisualGenRule — narrows CI when VisualGen source paths change.

VisualGen is a self-contained subsystem under
`tensorrt_llm/_torch/visual_gen/` (Flux / LTX-2 / Wan diffusion models)
plus the public API surface at `tensorrt_llm/visual_gen/`.

Block selection — entry-pattern based only:
VisualGen does NOT have its own `condition.terms.backend`; VG test
entries live in `backend: pytorch` and `backend: tensorrt` blocks.
A block "belongs to VG" iff any of its `tests:` entries matches one
of the three stable VG entry path families:
  - `unittest/_torch/visual_gen/...`         (28 entries)
  - `examples/test_visual_gen.py...`         (1 entry)
  - `visual_gen/test_visual_gen_benchmark.py` (1 entry)

Outward-facing fallback:
Unlike AutoDeploy, VG is imported eagerly (module-level) by non-VG
code: `commands/serve.py`, `commands/utils.py`, and
`serve/openai_server.py` import `VisualGenArgs` / `ParallelConfig` /
`VisualGen` / `VisualGenParams` at top level. A signature change to
those symbols can break trtllm-serve startup, which would affect
non-VG tests. The 5 files that define / re-export those symbols are
listed in `_VG_OUTWARD_FILES`; touching any of them forces fallback
even if the rest of the diff is VG-internal.
"""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex, _entry_target

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult

# VG source-path prefixes the rule may claim. Tests under tests/** are
# left to TestsDefRule; the two rules' scopes combine via
# _TESTSONLY_FAMILY.
_VG_SRC_PREFIXES: tuple[str, ...] = (
    "examples/visual_gen/",
    "tensorrt_llm/_torch/visual_gen/",
    "tensorrt_llm/visual_gen/",
)

# Files inside _VG_SRC_PREFIXES that are imported eagerly by non-VG
# code (top-level `from ... import VisualGenArgs / ParallelConfig /
# VisualGen / VisualGenParams`). Touching any of these can break
# trtllm-serve / trtllm-bench startup paths, so the rule defers to
# baseline rather than narrowing.
_VG_OUTWARD_FILES: frozenset[str] = frozenset(
    {
        "tensorrt_llm/_torch/visual_gen/config.py",
        "tensorrt_llm/visual_gen/__init__.py",
        "tensorrt_llm/visual_gen/args.py",
        "tensorrt_llm/visual_gen/params.py",
        "tensorrt_llm/visual_gen/visual_gen.py",
    }
)

# Substrings that mark a test entry as VG. Cover all three path
# families that appear in test-db YAMLs (audited 2026-05).
_VG_ENTRY_PATTERNS: tuple[str, ...] = (
    "unittest/_torch/visual_gen/",
    "examples/test_visual_gen.py",
    "visual_gen/test_visual_gen_benchmark.py",
)


def _is_vg_claim(path: str) -> bool:
    """Decide whether VisualGenRule claims `path`.

    `*.md` files are excluded so docs-only PRs (e.g.
    `examples/visual_gen/README.md`) don't force VG stages —
    `OutOfScopeRule` claims them as noop instead. Image suffixes are
    intentionally NOT excluded: VG ships reference images that are
    loaded as test fixtures (e.g. `examples/visual_gen/cat_piano.png`
    referenced by `tests/unittest/_torch/visual_gen/`), so edits to
    them must still force VG stages.
    """
    if not path.startswith(_VG_SRC_PREFIXES):
        return False
    if path.endswith(".md"):
        return False
    return True


def _entry_is_vg(entry: str) -> bool:
    return any(p in entry for p in _VG_ENTRY_PATTERNS)


def _vg_entries(block) -> list[str]:
    return [t for t in block.tests if _entry_is_vg(t)]


def _is_perf_stem(stem: str) -> bool:
    return stem == "l0_perf" or "perf_sanity" in stem


class VisualGenRule(Rule):
    name = "visualgen"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if _is_vg_claim(f)}
        if not claimed:
            return None

        # Outward-facing VG files break the "self-contained subsystem"
        # assumption — they are imported eagerly by trtllm-serve /
        # trtllm-bench. Claim the files (so they don't go unhandled and
        # silently fallback) but emit scope=None so Selector falls back
        # to baseline coverage instead of narrowing to VG-only stages.
        outward = claimed & _VG_OUTWARD_FILES
        if outward:
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"visualgen: {len(outward)} outward-facing VG file(s) "
                    f"touched ({sorted(outward)[0]}{'...' if len(outward) > 1 else ''}); "
                    "fallback to baseline"
                ),
            )

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            entries = _vg_entries(block)
            if not entries:
                continue
            key = (block.yaml_stem, block.block_index)
            prefix_dict = block_filters.setdefault(key, {})
            for entry in entries:
                target = _entry_target(entry)
                if target:
                    prefix_dict.setdefault(target, set()).add(entry)

        if not block_filters:
            # Defensive: VG source changed but no VG blocks exist in any
            # yaml. Don't fabricate stages — fall back to baseline so
            # the change still gets coverage.
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"visualgen: {len(claimed)} VG source file(s); "
                    "no VG block matched in any test-db yaml — fallback"
                ),
            )

        affected = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        perfsanity_relevant = any(_is_perf_stem(stem) for stem, _ in block_filters)

        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="visualgenonly",
            block_filters=block_filters,
            sanity_relevant=False,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"visualgen: {len(claimed)} VG source file(s) → "
                f"{len(block_filters)} VG block(s), {len(affected)} stage(s)"
            ),
        )
