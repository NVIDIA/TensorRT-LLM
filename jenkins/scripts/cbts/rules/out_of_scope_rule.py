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
"""OutOfScopeRule — claims paths that have no pre-merge CI impact.

Emits scope="noop" with empty stages and sanity / perfsanity off, so when
this is the only rule that fires, no test stage runs (Build still runs
upstream of CBTS Layer 2). When other rules also fire, the "noop" scope
gives way in `_combine_scopes` to the actionable scope.

`is_out_of_scope` is the single source of truth for these path patterns;
`TestsDefRule` imports it to skip the same paths, so out-of-scope claims
are not overridden by a same-file narrow contribution.
"""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex

from .base import PRInputs, Rule, RuleResult

# Path prefixes whose changes neither pre-merge nor post-merge L0 consumes.
# - tests/integration/test_lists/qa/   : QA test lists, run by separate
#   nightly QA workflows.
# - tests/integration/test_lists/dev/  : developer-side artifacts, not
#   consumed by any L0 pipeline.
# - tests/integration/defs/.test_durations : pytest-split timing cache;
#   used at runtime, doesn't affect test selection.
# - tests/integration/defs/agg_unit_mem_df.csv : per-(gpu, case)
#   pytest-xdist parallel_factor table consumed by test_unittests.py;
#   tunes worker count only, no impact on which tests run or their
#   results.
# - tests/microbenchmarks/             : benchmarking scripts, not run
#   by any L0 stage.
# - jenkins/scripts/cbts/              : CBTS itself — rules, main, and
#   debug tools that run at decision time. Edits change narrowing logic
#   for future PRs but cannot affect the outcome of any test stage in
#   the current PR.
OUT_OF_SCOPE_PREFIXES: tuple[str, ...] = (
    "tests/integration/test_lists/qa/",
    "tests/integration/test_lists/dev/",
    "tests/integration/defs/.test_durations",
    "tests/integration/defs/agg_unit_mem_df.csv",
    "tests/microbenchmarks/",
    "jenkins/scripts/cbts/",
)

# Path suffixes (extensions) under tests/ with no test-execution impact.
OUT_OF_SCOPE_TESTS_SUFFIXES: tuple[str, ...] = (".md",)


def is_out_of_scope(path: str) -> bool:
    if any(path.startswith(p) for p in OUT_OF_SCOPE_PREFIXES):
        return True
    if path.startswith("tests/") and path.endswith(OUT_OF_SCOPE_TESTS_SUFFIXES):
        return True
    return False


class OutOfScopeRule(Rule):
    name = "outofscope"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        # Stored for parity with other rules' constructor shape; not used.
        self.yaml_index = yaml_index

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if is_out_of_scope(f)}
        if not claimed:
            return None
        return RuleResult(
            handled_files=claimed,
            affected_stages=set(),
            scope="noop",
            sanity_relevant=False,
            perfsanity_relevant=False,
            reason=f"outofscope: {len(claimed)} file(s) match no-pre-merge-impact pattern",
        )
