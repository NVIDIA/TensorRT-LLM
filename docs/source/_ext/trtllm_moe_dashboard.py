# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sphinx directive that renders the MoE Perf Dashboard as an in-page widget.

The dashboard markup is an empty skeleton; ``moe_dashboard.js`` fills it in from the
single-version payload in ``_static/moe-perf-dashboard/data.js``. Assets are attached
only to the page that uses the directive, because the data file is a payload every
other page would download for nothing.
"""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive

#: Directory under ``_static`` holding the vendored dashboard payload
#: (``config.js`` and the single-version ``data.js``).
PAYLOAD_DIR = "moe-perf-dashboard"

#: Scripts to attach, in load order: config and data define the globals that
#: ``moe_dashboard.js`` consumes, so they must come first.
PAGE_SCRIPTS = (
    f"{PAYLOAD_DIR}/config.js",
    f"{PAYLOAD_DIR}/data.js",
    "moe_dashboard.js",
)

PAGE_STYLES = ("moe_dashboard.css",)

#: Element ids are namespaced with ``moe-`` so they cannot collide with ids the
#: documentation theme owns; ``moe_dashboard.js`` prepends the prefix in one place.
SKELETON = """\
<div id="moe-dashboard">
  <div class="wrap">
    <div id="moe-banner" class="banner" hidden></div>
  </div>
  <div class="moe-main">
    <div class="wrap stage" id="moe-stage">
      <div id="moe-filters" class="filters" aria-label="Scenario filters"></div>
      <div class="content">
        <div class="status-bar">
          <p id="moe-status" class="status" role="status"></p>
        </div>
        <section id="moe-results" class="results" aria-live="polite"></section>
      </div>
    </div>
  </div>
</div>
"""


class TRTLLMMoEDashboard(Directive):
    """Embed the interactive MoE Perf Dashboard widget."""

    has_content = False
    option_spec = {}

    def run(self):
        env = self.state.document.settings.env
        # Recorded so ``_on_html_page_context`` knows which page needs the assets.
        env.metadata[env.docname]["trtllm_moe_dashboard"] = True
        return [nodes.raw("", SKELETON, format="html")]


def _on_html_page_context(app, pagename, templatename, context, doctree):
    """Attach the dashboard assets to the page carrying the directive, only."""
    if not app.env.metadata.get(pagename, {}).get("trtllm_moe_dashboard"):
        return

    for css in PAGE_STYLES:
        app.add_css_file(css)
    for js in PAGE_SCRIPTS:
        app.add_js_file(js)


def setup(app):
    app.add_directive("trtllm_moe_dashboard", TRTLLMMoEDashboard)
    app.connect("html-page-context", _on_html_page_context)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
