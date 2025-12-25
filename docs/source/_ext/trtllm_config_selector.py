# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


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


def _ensure_repo_root_on_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _write_config_db_json(app) -> None:
    builder = getattr(app, "builder", None)
    if not builder:
        return
    if builder.name not in {"html", "dirhtml"}:
        return

    _ensure_repo_root_on_syspath()
    from examples.configs.database.database import DATABASE_LIST_PATH
    from scripts.generate_config_table import generate_json

    out_static = Path(builder.outdir) / "_static"
    out_static.mkdir(parents=True, exist_ok=True)
    out_path = out_static / "config_db.json"
    generate_json(Path(DATABASE_LIST_PATH), output_file=out_path)
    LOGGER.info("Wrote config selector database: %s", out_path)


def _on_build_finished(app, exception) -> None:
    if exception is not None:
        return
    _write_config_db_json(app)


def setup(app):
    app.add_css_file("config_selector.css")
    app.add_js_file("config_selector.js")
    app.add_directive("trtllm_config_selector", TRTLLMConfigSelector)
    # Generate config_db.json into the HTML output _static directory at build time.
    app.connect("build-finished", _on_build_finished)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
