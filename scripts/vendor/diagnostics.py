# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actionable diagnostics safe to publish separately from operator failures."""

from __future__ import annotations

import html
from typing import Literal

import yaml


class Diagnostic(ValueError):
    """A recognized problem with a corrective action and an optional Markdown example."""

    def __init__(
        self,
        problem: str,
        action: str,
        *,
        example: str = "",
        owner: Literal["author", "maintainer"] = "author",
        notification_text: str | None = None,
    ) -> None:
        self.problem, self.action, self.example, self.owner = problem, action, example, owner
        # Observation-only changes must not notify authors about the same actionable result.
        self.notification_text = notification_text
        super().__init__(problem)

    def __str__(self) -> str:
        return f"{self.problem}\n\n**Fix:** {self.action}" + (
            f"\n\n{self.example}" if self.example else ""
        )


def field(value: object) -> str:
    """Quote a bounded field identifier without interpreting author-supplied Markdown."""
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")[:160]
    return "<code>" + html.escape(text) + "</code>"


def yaml_problem(error: yaml.YAMLError) -> str:
    """Describe YAML failures and locations without echoing source excerpts or values."""
    detail = "invalid YAML syntax or unsupported YAML value"
    if isinstance(error, yaml.MarkedYAMLError):
        problem = error.problem or ""
        if problem.startswith("found duplicate key"):
            detail = "duplicate mapping key; keep one value for each key"
        elif "merge keys are not allowed" in problem:
            detail = "YAML merge keys are not allowed; write fields explicitly"
        elif problem == "found an unhashable mapping key":
            detail = "mapping keys must be scalar field names, not lists or mappings"
        elif "could not find expected ':'" in problem:
            detail = "missing ':' after a mapping key"
        elif problem.startswith("expected ',' or ']'"):
            detail = "expected a comma between list items or a closing ']'"
        elif problem.startswith("expected ',' or '}'"):
            detail = "expected a comma between mapping entries or a closing '}'"
        elif problem.startswith("expected <block end>"):
            detail = "unexpected field or indentation; align sibling fields with spaces"
        mark = error.problem_mark
        if mark is not None:
            detail += f" at YAML line {mark.line + 1}, column {mark.column + 1}"
    return detail
