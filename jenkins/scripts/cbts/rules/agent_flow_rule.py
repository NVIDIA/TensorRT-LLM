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
"""AgentFlowRule — narrows CI when agent-flow source paths change.

``agent-flow/`` is a self-contained sub-project (own ``pyproject.toml``,
own dependencies, pure-CPU pytest suite) with a single dedicated CI stage
``CPU-AgentFlow-UnitTest``. Nothing outside ``agent-flow/`` imports it, and it
imports nothing from the TRT-LLM wheel, so an agent-flow-only change needs
exactly that one CPU stage and none of the GPU test-db stages.

Unlike the AutoDeploy / VisualGen rules, this stage is not driven by any
test-db YAML, so there are no blocks to resolve — the rule contributes the
stage name literally. ``AGENT_FLOW_STAGE`` MUST stay in sync with the
matching stage key in ``jenkins/L0_Test.groovy`` (``agentFlowTestConfigs``);
if they diverge, CBTS Layer 2 keeps nothing and the stage silently stops
running.
"""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex

from .base import PRInputs, Rule, RuleResult

# Stage key as declared in jenkins/L0_Test.groovy. Keep in sync.
AGENT_FLOW_STAGE = "CPU-AgentFlow-UnitTest"

# Every changed file under this prefix is claimed by the rule.
_AGENT_FLOW_PREFIX = "agent-flow/"


def _is_agent_flow_claim(path: str) -> bool:
    """Decide whether AgentFlowRule claims ``path``.

    Claims everything under ``agent-flow/`` — source, tests, and build
    metadata (``pyproject.toml``, ``.pre-commit-config.yaml``) all affect
    what the pytest stage installs and runs — except ``*.md`` docs, which
    ``OutOfScopeRule`` claims as noop so a docs-only edit doesn't force the
    stage.
    """
    if not path.startswith(_AGENT_FLOW_PREFIX):
        return False
    if path.endswith(".md"):
        return False
    return True


class AgentFlowRule(Rule):
    name = "agentflow"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        # Stored for parity with other rules' constructor shape; not used —
        # the agent-flow stage is not test-db-driven.
        self.yaml_index = yaml_index

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {f for f in pr.changed_files if _is_agent_flow_claim(f)}
        if not claimed:
            return None

        return RuleResult(
            handled_files=claimed,
            affected_stages={AGENT_FLOW_STAGE},
            scope="agentflowonly",
            # The agent-flow stage builds no wheel and runs no perf benchmark,
            # so neither the package-sanity nor the perf-sanity carve-outs
            # apply to an agent-flow-only change.
            sanity_relevant=False,
            perfsanity_relevant=False,
            reason=(f"agentflow: {len(claimed)} agent-flow file(s) → 1 stage ({AGENT_FLOW_STAGE})"),
        )
