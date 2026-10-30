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
"""VisualGenRule — narrows CI when VisualGen source paths change.

VisualGen is a self-contained subsystem under
`tensorrt_llm/_torch/visual_gen/` (Flux / LTX-2 / Wan diffusion models)
plus the public API surface at `tensorrt_llm/visual_gen/` and the media
helpers at `tensorrt_llm/media/`.

Block selection — entry-pattern based only:
VisualGen does NOT have its own `condition.terms.backend`; VG test
entries live in `backend: pytorch` and `backend: tensorrt` blocks.
A block "belongs to VG" iff any of its `tests:` entries lives under a
dedicated `visual_gen/` test path or is the VisualGen perf-sanity entry.

Shared serving boundary:
Public VisualGen and media modules participate in shared serving imports.
Their changes keep the complete CPU suite, shared serving integration tests,
and package/performance sanity checks in addition to VG entries. Missing API,
schema, lazy-import, or serving validation keeps the baseline fallback.
"""

from __future__ import annotations

import re
from typing import Optional

from cbts.blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree

from ._helpers import (
    block_filters_for_stem,
    is_perf_stem,
    resolve_affected_stages,
    stages_by_yaml_stem,
)
from .base import PRInputs, Rule, RuleResult
from .docs import CPU_TEST_YAML_STEM, is_docs_path

# VG source-path prefixes the rule may claim, mirroring the VisualGen
# section of `.github/CODEOWNERS`. Tests under tests/** are left to
# TestsDefRule; the two rules' scopes combine via _TESTSONLY_FAMILY.
# fused-DiT paths stay unclaimed: those diffs also touch `inplace_info()`
# in `tensorrt_llm/_torch/compilation/utils.py`, which no VG rule claims.
_VG_SRC_PREFIXES: tuple[str, ...] = (
    "examples/visual_gen/",
    "scripts/visualgen_eval/",
    "tensorrt_llm/_torch/visual_gen/",
    "tensorrt_llm/media/",
    "tensorrt_llm/visual_gen/",
)

_VG_SCHEMA_PATH = "tensorrt_llm/schemas/trtllm-serve-visual-gen-config.schema.json"
_SCHEMA_TEST_TARGET = "unittest/llmapi/test_trtllm_serve_config_schemas.py"

# These packages expose shared protocol and media dependencies to serving.
_VG_OUTWARD_PREFIXES: tuple[str, ...] = (
    "tensorrt_llm/media/",
    "tensorrt_llm/visual_gen/",
)

# Keep complete serving families: protocol/media consumers include normal,
# disaggregated, multimodal, and benchmark requests, not just server startup.
_SHARED_SERVING_PREFIXES = (
    "test_e2e.py",
    "examples/serve",
    "disaggregated",
    "llmapi",
    "unittest/llmapi",
    "unittest/disaggregated",
)
_PUBLIC_CPU_CHECKS = (
    "unittest/api_stability",
    _SCHEMA_TEST_TARGET,
    "unittest/llmapi/apps/test_serve_lazy_import.py",
    "unittest/others/test_lazy_model_zoo.py",
)
_PUBLIC_SERVING_CHECKS = (
    "test_e2e.py::test_openai_chat_example",
    "test_e2e.py::test_openai_completions_example",
    "examples/serve/test_serve.py::test_config_override_precedence",
)


def _covers_check(entry: str, check: str) -> bool:
    """Require an unfiltered test entry covering the complete validation target."""
    options = entry.split()[1:]
    if any(
        option == "SKIP"
        or option.split("=", 1)[0] in {"-k", "-m", "--ignore", "--ignore-glob", "--deselect"}
        for option in options
    ):
        return False
    target = _entry_target(entry)
    return bool(target and _target_in_filter_subtree(check, target))


# Substrings that mark a test entry as VG. VG tests are expected to live
# under dedicated visual_gen test directories, except the perf-sanity
# frontend which stays with the shared perf tests.
_VG_ENTRY_PATTERNS: tuple[str, ...] = (
    "visual_gen/",
    "perf/test_visual_gen_perf_sanity.py",
)


def _is_vg_claim(path: str) -> bool:
    """Decide whether VisualGenRule claims `path`.

    Documentation files are excluded so docs-only PRs (e.g.
    `examples/visual_gen/README.md`) don't force VG stages; `DocsRule`
    routes them to the docs build instead. Image suffixes are
    intentionally NOT excluded: VG ships reference images that are
    loaded as test fixtures (e.g. `examples/visual_gen/cat_piano.png`
    referenced by `tests/unittest/_torch/visual_gen/`), so edits to
    them must still force VG stages.
    """
    if path == _VG_SCHEMA_PATH:
        return True
    if not path.startswith(_VG_SRC_PREFIXES):
        return False
    if is_docs_path(path):
        return False
    return True


def _entry_is_vg(entry: str) -> bool:
    return any(p in entry for p in _VG_ENTRY_PATTERNS)


def _vg_entries(block) -> list[str]:
    return [t for t in block.tests if _entry_is_vg(t)]


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

        outward = {f for f in claimed if f.startswith(_VG_OUTWARD_PREFIXES)}

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        schema_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            entries = _vg_entries(block)
            if _VG_SCHEMA_PATH in claimed:
                for entry in block.tests:
                    target = _entry_target(entry)
                    if target and (
                        target == _SCHEMA_TEST_TARGET
                        or _SCHEMA_TEST_TARGET.startswith(target.rstrip("/") + "/")
                    ):
                        key = (block.yaml_stem, block.block_index)
                        schema_filters.setdefault(key, {}).setdefault(target, set()).add(entry)
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
        if outward:
            shared_filters = block_filters_for_stem(self.yaml_index, CPU_TEST_YAML_STEM)
            validation_stages = {
                stem: [
                    (name, stage)
                    for name, stage in stages
                    if "-OnDemand-" not in name
                    and (
                        pr.post_merge
                        or ("Post-Merge" not in name and not re.search(r"\d+_GPUs", name))
                    )
                ]
                for stem, stages in self._stages_by_yaml.items()
            }
            required = set(_PUBLIC_CPU_CHECKS) | set(_PUBLIC_SERVING_CHECKS)
            for check in sorted(required):
                check_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
                for block in self.yaml_index.blocks:
                    if check in _PUBLIC_CPU_CHECKS and block.yaml_stem != CPU_TEST_YAML_STEM:
                        continue
                    for entry in block.tests:
                        if _covers_check(entry, check):
                            key = (block.yaml_stem, block.block_index)
                            check_filters.setdefault(key, {}).setdefault(
                                _entry_target(entry), set()
                            ).add(entry)
                if not resolve_affected_stages(check_filters, self.yaml_index, validation_stages):
                    return RuleResult(
                        handled_files=claimed,
                        affected_stages=set(),
                        scope=None,
                        reason=f"visualgen: required shared validation unavailable: {check}",
                    )
            for block in self.yaml_index.blocks:
                for entry in block.tests:
                    target = _entry_target(entry)
                    if any(
                        _target_in_filter_subtree(target, prefix)
                        for prefix in _SHARED_SERVING_PREFIXES
                    ):
                        key = (block.yaml_stem, block.block_index)
                        shared_filters.setdefault(key, {}).setdefault(target, set()).add(entry)
            for key, prefixes in shared_filters.items():
                for target, entries in prefixes.items():
                    block_filters.setdefault(key, {}).setdefault(target, set()).update(entries)
            affected |= resolve_affected_stages(
                shared_filters, self.yaml_index, self._stages_by_yaml
            )
        if _VG_SCHEMA_PATH in claimed:
            schema_stages = resolve_affected_stages(
                schema_filters, self.yaml_index, self._stages_by_yaml
            )
            if not schema_stages:
                return RuleResult(
                    handled_files=claimed,
                    affected_stages=set(),
                    scope=None,
                    reason="visualgen: schema validation stages unavailable — fallback",
                )
            affected |= schema_stages
            for key, prefixes in schema_filters.items():
                for target, entries in prefixes.items():
                    block_filters.setdefault(key, {}).setdefault(target, set()).update(entries)
        perfsanity_relevant = bool(outward) or any(is_perf_stem(stem) for stem, _ in block_filters)

        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="visualgenonly",
            block_filters=block_filters,
            sanity_relevant=bool(outward),
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"visualgen: {len(claimed)} VG source file(s) → "
                f"{len(block_filters)} block(s), {len(affected)} stage(s)"
                + ("; shared serving + complete CPU validation" if outward else "")
            ),
        )
