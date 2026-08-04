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
"""SpecDecRule — narrows CI when speculative-decoding source paths change.

The speculative-decoding subsystem lives in `tensorrt_llm/_torch/speculative/`
plus per-drafter model dirs (`tensorrt_llm/models/{eagle,medusa,redrafter}/`)
and the matching `examples/` references.

Block selection — entry-pattern based only:
Spec-dec has no `condition.terms.backend` of its own; entries live in
`backend: pytorch` and `backend: tensorrt` blocks. A block "belongs to
spec-dec" iff any of its `tests:` entries matches a stable spec-dec
marker — see `_SPEC_ENTRY_PATTERNS` for the full set, validated against
the May 2026 test-db.

Outward fallback: not needed. The only non-spec-dec eager import of
spec-dec types is `tensorrt_llm/commands/build.py` pulling
`SpeculativeDecodingMode` from `tensorrt_llm/models/modeling_utils.py`,
which lives outside `_SPEC_SRC_PREFIXES` and therefore is never claimed
by this rule. Such PRs naturally fall back to baseline.

PerfSanity policy: `perfsanity_relevant` is dynamic — True only when a
matched block lives in a `*_perf_sanity*` yaml (mirrors AutoDeployRule /
VisualGenRule). Spec-dec PRs whose entry matches don't land in any
perf-sanity yaml will have `perfsanity_required=False` aggregated,
letting Groovy Layer 2 drop the force-keep of `*-PerfSanity-*` stages.
PRs whose entries do reach perf-sanity blocks keep those stages.
"""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex, _entry_target

from ._helpers import resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult

# Spec-dec source-path prefixes the rule may claim. Tests under tests/**
# are left to TestsDefRule; the two rules' scopes combine via
# _TESTSONLY_FAMILY.
_SPEC_SRC_PREFIXES: tuple[str, ...] = (
    "tensorrt_llm/_torch/speculative/",
    "tensorrt_llm/models/eagle/",
    "tensorrt_llm/models/medusa/",
    "tensorrt_llm/models/redrafter/",
    "examples/eagle/",
    "examples/medusa/",
    "examples/redrafter/",
    "examples/draft_target_model/",
    "examples/ngram/",
    "examples/llm-api/llm_speculative_decoding.py",
)

# Stable substrings that mark a test entry as spec-dec. Validated against
# tests/integration/test_lists/test-db/ (May 2026):
#   - "test_eagle"          → test_eagle.py + test_eagle3* + *_eagle*
#                             param ids in test_llm_api_pytorch.py and
#                             test_e2e.py (~30 entries)
#   - "test_medusa"         → examples/test_medusa.py (~9 entries)
#   - "test_redrafter"      → examples/test_redrafter.py (~9 entries)
#   - "test_ngram"          → examples/test_ngram.py (~2 entries)
#   - "test_draft_target_model" → ~4 entries
#   - "test_ad_speculative_decoding" → AD-backed spec-dec tests (~6 entries)
#   - "unittest/_torch/speculative/" → ~3 entries
#   - "test_spec_decoding_metrics"   → serve metrics tests
#   - "test_llmapi_speculative_decoding" → ~3 entries
#   - "speculative_decoding_bls"     → triton BLS spec-dec test (1 entry)
#   - "test_mtp"            → method-name match: TestNemotronSuperV3 MTP +
#                             KVCacheV2DSv3Lite MTP scheduler tests
#                             (~5 entries)
#   - "mtp_nextn"           → DeepSeek mtp_nextn=N parametrize id
#                             (~146 entries with N>0; the =0 case is
#                             filtered in `_entry_is_spec`)
#   - "_mtp"                → throughput_mtp / *_mtp1 / *_mtp3 / *_mtp_*
#                             style parametrize suffixes (~50 entries)
_SPEC_ENTRY_PATTERNS: tuple[str, ...] = (
    "test_eagle",
    "test_medusa",
    "test_redrafter",
    "test_ngram",
    "test_draft_target_model",
    "test_ad_speculative_decoding",
    "unittest/_torch/speculative/",
    "test_spec_decoding_metrics",
    "test_llmapi_speculative_decoding",
    "speculative_decoding_bls",
    "test_mtp",
    "mtp_nextn",
    "_mtp",
)

# mtp_nextn=0 disables MTP for that parametrization — those entries
# test baseline (no spec-dec) behavior and should not be claimed by
# bare `mtp_nextn` substring matching. See `_entry_is_spec`.
_MTP_DISABLED_MARKER = "mtp_nextn=0"


def _is_spec_claim(path: str) -> bool:
    """Decide whether SpecDecRule claims `path`.

    `*.md` files are excluded so docs-only PRs (e.g.
    `examples/eagle/README.md`) don't force spec-dec stages —
    `OutOfScopeRule` claims them as noop instead. Other suffixes
    (`.png` / `.jpg` / etc.) are NOT excluded here: a binary asset
    under a spec-dec path could be a test fixture, so the rule keeps
    claiming them and forces spec-dec stages to re-run (safe over-run).
    """
    if not path.startswith(_SPEC_SRC_PREFIXES):
        return False
    if path.endswith(".md"):
        return False
    return True


def _entry_is_spec(entry: str) -> bool:
    """Return True iff `entry` is a spec-dec test entry.

    `mtp_nextn=0` carve-out: those parametrizations test MTP-disabled
    baseline behavior, not the spec-dec code path. Skip them unless the
    entry also carries another spec-dec marker (e.g. `_mtp`, `test_eagle`).
    """
    if _MTP_DISABLED_MARKER in entry:
        return any(p in entry for p in _SPEC_ENTRY_PATTERNS if p != "mtp_nextn")
    return any(p in entry for p in _SPEC_ENTRY_PATTERNS)


def _spec_entries(block) -> list[str]:
    return [t for t in block.tests if _entry_is_spec(t)]


def _is_perf_sanity_stem(stem: str) -> bool:
    """True for perf-sanity yaml stems (`l0_*_perf_sanity*`)."""
    return "perf_sanity" in stem


class SpecDecRule(Rule):
    name = "specdec"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if _is_spec_claim(f)}
        if not claimed:
            return None

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for block in self.yaml_index.blocks:
            entries = _spec_entries(block)
            if not entries:
                continue
            key = (block.yaml_stem, block.block_index)
            prefix_dict = block_filters.setdefault(key, {})
            for entry in entries:
                target = _entry_target(entry)
                if target:
                    prefix_dict.setdefault(target, set()).add(entry)

        if not block_filters:
            # Defensive: spec-dec source changed but no spec-dec blocks
            # exist in any yaml. Don't fabricate stages — fall back to
            # baseline so the change still gets coverage.
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"specdec: {len(claimed)} spec-dec source file(s); "
                    "no spec-dec block matched in any test-db yaml — fallback"
                ),
            )

        affected = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        perfsanity_relevant = any(_is_perf_sanity_stem(stem) for stem, _ in block_filters)

        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="specdeconly",
            block_filters=block_filters,
            sanity_relevant=False,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"specdec: {len(claimed)} spec-dec source file(s) → "
                f"{len(block_filters)} spec-dec block(s), {len(affected)} stage(s)"
            ),
        )
