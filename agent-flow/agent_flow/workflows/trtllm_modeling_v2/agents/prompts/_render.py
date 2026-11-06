# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render Markdown prompts from versioned package resources."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined

_ENVIRONMENT = Environment(
    loader=PackageLoader(__package__, "templates"),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


def render_template(template_name: str, **context: object) -> str:
    """Render a packaged template, failing on missing context instead of omitting it."""
    return _ENVIRONMENT.get_template(template_name).render(**context)


__all__ = ["render_template"]
