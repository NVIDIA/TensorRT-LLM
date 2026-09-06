"""Extract the planner's ``## Execution Graph`` block from ``plan.md``.

This is the workflow-layer glue that turns the DAG the planner declares in
``plan.md`` into an :class:`~agent_flow.orchestration.ExecutionGraph` the engine
can run. It does *only* the markdown scan — locating the ``## Execution Graph``
heading and the first fenced code block inside that section — and hands the
block body to :func:`~agent_flow.orchestration.parse_execution_graph`, which owns
all schema and structural validation (Phase 1). No graph rules are
re-implemented here.

Contract:

- No ``## Execution Graph`` section anywhere => ``None`` (a legacy/linear plan;
  the caller falls back to the single-cursor path).
- Section present but its fenced block is missing, unterminated, or empty =>
  :class:`~agent_flow.orchestration.ExecutionGraphError`. A malformed graph must
  fail loudly, never be silently treated as "no graph".
- Section + block present but the YAML is invalid (cycle, unknown dep, duplicate
  id, bad enum) => the parser's ``ExecutionGraphError`` propagates unchanged.
"""

from __future__ import annotations

import re

from agent_flow.orchestration import ExecutionGraph, ExecutionGraphError, parse_execution_graph

__all__ = ["extract_execution_graph"]

# The section heading the planner writes: a level-2 ``## `` heading ending in
# "Execution Graph", tolerating flexible inner/trailing whitespace AND an
# optional leading section number (planners routinely number every section, so
# ``## 7. Execution Graph`` / ``## 3) Execution Graph`` must match too — an
# exact-only match silently drops a perfectly good graph and aborts --concurrent).
_HEADING_RE = re.compile(r"##\s+(?:\d+[.)]?\s+)?Execution\s+Graph\s*$")

# The opening fence of a code block: three backticks, optionally followed by a
# language tag (e.g. ``yaml``). The closing fence is bare three backticks.
_FENCE = "```"


def _is_execution_graph_heading(line: str) -> bool:
    return _HEADING_RE.fullmatch(line.strip()) is not None


def _is_section_boundary(line: str) -> bool:
    """True for a following ``## `` (level-2) heading that ends the section.

    A deeper ``### `` sub-heading is *not* a boundary — ``startswith("## ")``
    excludes it because its third character is ``#`` rather than a space.
    """
    return line.lstrip().startswith("## ")


def extract_execution_graph(plan_md_text: str) -> ExecutionGraph | None:
    """Extract and parse the ``## Execution Graph`` block from plan markdown.

    Args:
        plan_md_text: The full text of ``plan.md``.

    Returns:
        The parsed :class:`ExecutionGraph`, or ``None`` when the plan has no
        ``## Execution Graph`` section at all.

    Raises:
        ExecutionGraphError: When the section exists but its fenced block is
            missing, unterminated, or empty, or when the block's YAML is invalid
            (the latter propagated verbatim from ``parse_execution_graph``).
    """
    lines = plan_md_text.splitlines()

    heading_index = next(
        (i for i, line in enumerate(lines) if _is_execution_graph_heading(line)),
        None,
    )
    if heading_index is None:
        # No section at all: a legacy/linear plan. The caller uses the
        # single-cursor path — this is the one case that is *not* an error.
        return None

    # Scan the section body for the opening fence, stopping at the next ``## ``
    # heading (a fence that belongs to a later section must not be adopted).
    open_index: int | None = None
    for i in range(heading_index + 1, len(lines)):
        if _is_section_boundary(lines[i]):
            break
        if lines[i].strip().startswith(_FENCE):
            open_index = i
            break

    if open_index is None:
        raise ExecutionGraphError(
            "'## Execution Graph' section has no fenced code block; "
            "a declared graph must be a fenced YAML block"
        )

    # Collect the block body up to the closing bare-fence line. A ``## `` line
    # inside the block (e.g. a YAML comment) is body, not a boundary, so the
    # closing fence — not the next heading — terminates the scan.
    close_index: int | None = None
    for i in range(open_index + 1, len(lines)):
        if lines[i].strip() == _FENCE:
            close_index = i
            break

    if close_index is None:
        raise ExecutionGraphError(
            "'## Execution Graph' fenced code block is unterminated (missing closing ```)"
        )

    body = "\n".join(lines[open_index + 1 : close_index])
    if not body.strip():
        raise ExecutionGraphError("'## Execution Graph' fenced code block is empty")

    # Delegate every schema/structural check (unknown dep, cycle, duplicate id,
    # bad enum, malformed YAML) to the engine parser; let its error propagate.
    return parse_execution_graph(body)
