# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive, directives


class TRTLLMConfigSelector(Directive):
    """Embed the interactive config selector widget.

    Usage:
        .. trtllm_config_selector::
           :models: deepseek-ai/DeepSeek-R1-0528, nvidia/DeepSeek-R1-0528-FP4-v2
           :config_db: _static/config_db.json
    """

    has_content = False
    option_spec = {
        # Comma-separated list of HF model ids to include (optional).
        "models": directives.unchanged,
        # Path relative to doc root (optional).
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
    # Static assets live under docs/source/_static/.
    app.add_css_file("config_selector.css")
    app.add_js_file("config_selector.js")
    app.add_directive("trtllm_config_selector", TRTLLMConfigSelector)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
