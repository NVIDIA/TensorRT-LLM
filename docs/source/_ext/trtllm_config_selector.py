# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive, directives


class TRTLLMConfigSelector(Directive):
    """Embed the interactive config selector widget."""

    has_content = False
    option_spec = {
        "models": directives.unchanged,
        "config_db": directives.unchanged,
    }

    def run(self):
        models = (self.options.get("models") or "").strip()
        config_db = (self.options.get("config_db") or "").strip()

        attrs = ['data-trtllm-config-selector="1"']
        if models:
            attrs.append(f'data-models="{models}"')
        if config_db:
            attrs.append(f'data-config-db="{config_db}"')

        html = f"<div {' '.join(attrs)}></div>"
        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_css_file("config_selector.css")
    app.add_js_file("config_selector.js")
    app.add_directive("trtllm_config_selector", TRTLLMConfigSelector)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
