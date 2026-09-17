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
"""DocsRule — routes documentation changes to the dedicated docs build."""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex

from .base import PRInputs, Rule, RuleResult

# Stage key declared in jenkins/L0_Test.groovy::docBuildConfigs. This stage is
# not test-db-driven, so CBTS contributes its name literally.
DOCS_STAGE = "CPU-Build_Docs"

_DOCS_PREFIX = "docs/"
_DOCS_SUFFIXES: tuple[str, ...] = (".md", ".rst")


def is_docs_path(path: str) -> bool:
    """Return whether ``path`` contributes to the repository documentation."""
    return path.startswith(_DOCS_PREFIX) or path.endswith(_DOCS_SUFFIXES)


class DocsRule(Rule):
    name = "docs"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        # Stored for parity with other rules' constructor shape; the docs build
        # is a dedicated stage rather than a test-db-backed stage.
        self.yaml_index = yaml_index

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed = {path for path in pr.changed_files if is_docs_path(path)}
        if not claimed:
            return None

        return RuleResult(
            handled_files=claimed,
            affected_stages={DOCS_STAGE},
            scope="docsonly",
            sanity_relevant=False,
            perfsanity_relevant=False,
            reason=f"docs: {len(claimed)} documentation file(s) → 1 stage ({DOCS_STAGE})",
        )
