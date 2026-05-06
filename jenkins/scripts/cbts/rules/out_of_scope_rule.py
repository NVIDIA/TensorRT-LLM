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
"""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex

from .base import PRInputs, Rule, RuleResult

# Path prefixes whose changes the pre-merge pipeline does not consume.
# - tests/integration/test_lists/qa/  : QA test lists, run by post-merge /
#   nightly QA workflows.
OUT_OF_SCOPE_PREFIXES: tuple[str, ...] = ("tests/integration/test_lists/qa/",)


class OutOfScopeRule(Rule):
    name = "outofscope"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        # Stored for parity with other rules' constructor shape; not used.
        self.yaml_index = yaml_index

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {
            f for f in pr.changed_files if any(f.startswith(p) for p in OUT_OF_SCOPE_PREFIXES)
        }
        if not claimed:
            return None
        return RuleResult(
            handled_files=claimed,
            affected_stages=set(),
            scope="noop",
            sanity_relevant=False,
            perfsanity_relevant=False,
            reason=f"outofscope: {len(claimed)} file(s) match no-pre-merge-impact prefix",
        )
