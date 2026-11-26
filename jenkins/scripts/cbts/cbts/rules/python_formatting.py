# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Recognize verified Python formatting changes without a coverage dependency."""

from __future__ import annotations

import ast
import io
import tokenize

from ._helpers import reconstruct_diff_pre_image
from .base import PRInputs, Rule, RuleResult


def is_formatting_only_change(post_source: str, diff: str) -> bool:
    """Compare complete syntax trees and comments after verifying the raw patch.

    Keep docstrings, type comments, and comment directives in the comparison.
    Missing source, malformed patches, and new or removed files remain unknown.
    """
    if any(
        line.startswith(
            (
                "new file mode ",
                "deleted file mode ",
                "old mode ",
                "new mode ",
                "rename from ",
                "rename to ",
                "copy from ",
                "copy to ",
            )
        )
        for line in diff.splitlines()
    ):
        return False
    pre_source = reconstruct_diff_pre_image(post_source, diff)
    if pre_source is None:
        return False
    try:
        before = ast.parse(pre_source, type_comments=True)
        after = ast.parse(post_source, type_comments=True)
        comments = [
            [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            for source in (pre_source, post_source)
        ]
    except (SyntaxError, ValueError, tokenize.TokenError):
        return False
    return (
        bool(before.body)
        and ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
        and comments[0] == comments[1]
    )


class PythonFormattingRule(Rule):
    name = "pythonformatting"
    needs_diff_for = ("**/*.py",)
    handles_docstrings = True

    def apply(self, pr: PRInputs) -> RuleResult | None:
        if not pr.formatting_only_files:
            return None
        return RuleResult(
            handled_files=set(pr.formatting_only_files),
            affected_stages=set(),
            scope="noop",
            reason=f"pythonformatting: {len(pr.formatting_only_files)} verified formatting-only file(s)",
            sanity_relevant=False,
            perfsanity_relevant=False,
        )
