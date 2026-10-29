"""Orchestrator-owned global observation window for the concurrent DAG.

Under ``--concurrent`` each node writes only into its own private
sub-workspace (``nodes/<slug>/status.md`` + ``progress.yaml``, Task 9), so the
parallel node agents never share writes to the top-level files. This module
gives a human one window into the whole run by rendering the top-level
``workspace/status.md`` and ``workspace/progress.yaml`` as DERIVED,
single-writer aggregations:

- :func:`render_global_status` overwrites ``status.md`` with a whole-DAG
  rollup (total nodes + counts by state) followed by one row per node
  (id, state, and the node's latest decision + one-line summary).
- :func:`render_global_progress` rewrites ``progress.yaml`` with a top-level
  ``timeline`` list that aggregates every node's private progress entries,
  each stamped with its ``node`` id and ordered by ``(timestamp, node)``,
  while PRESERVING the ``plan_stage`` / ``human_feedback`` sections the plan
  phase and ``--feedback`` wrote into that same shared file before the build.

Both never write into a node's sub-workspace and are idempotent — re-rendering
reproduces the same top-level file. ``render_global_status`` is a pure
derivation of the per-node files; ``render_global_progress`` additionally reads
back the top-level ``progress.yaml`` so it can carry the plan/feedback audit
trail through unchanged. A node whose private ``progress.yaml`` is empty or
missing yields a clean "no entries" row instead of crashing. The orchestrator
is the ONLY writer of these two top-level files; it wires both renderers into
``_run_concurrent_build`` after the scheduler run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .node_workspace import NODES_DIRNAME, PROGRESS_FILENAME, node_dir_slug
from .progress import BUILD_STAGE, HUMAN_FEEDBACK, PLAN_STAGE, read_progress
from .status import write_status_text

# Top-level (workspace-root) derived files owned by the orchestrator.
GLOBAL_STATUS_FILENAME = "status.md"
GLOBAL_PROGRESS_FILENAME = "progress.yaml"

# Display order for the rollup; any state not listed here (an unexpected or
# future ``NodeState`` value) is appended afterwards in sorted order so the
# rollup never silently drops a node's state.
_STATE_DISPLAY_ORDER = ("done", "running", "pending", "failed", "blocked", "interrupted")

# Agent stages carry the decisions/summaries a status row surfaces;
# ``human_feedback`` is the user's voice, not an agent decision, so it is
# excluded from the "latest decision" pick (but still shows in the timeline).
_AGENT_STAGES = (PLAN_STAGE, BUILD_STAGE)
_ALL_STAGES = (PLAN_STAGE, BUILD_STAGE, HUMAN_FEEDBACK)


def _node_progress_path(workspace: Path, node_id: str) -> Path:
    """Return the path to ``node_id``'s private ``progress.yaml`` (never created).

    Reuses :func:`node_dir_slug` so the id→slug rule is defined in exactly one
    place; this only computes the path — it never creates or seeds the file.
    """
    return workspace / NODES_DIRNAME / node_dir_slug(node_id) / PROGRESS_FILENAME


def _state_str(state: Any) -> str:
    """Normalize a node state to its plain string value.

    Accepts a :class:`~agent_flow.orchestration.NodeState` (a ``str`` enum,
    whose ``str()`` renders as ``NodeState.DONE``) or a plain string, and
    returns the underlying value (``"done"``). ``None`` becomes ``"unknown"``.
    """
    if state is None:
        return "unknown"
    return str(getattr(state, "value", state))


def _one_line(text: Any) -> str:
    """Collapse a (possibly multi-line) summary to its first non-empty line."""
    stripped = (str(text) if text is not None else "").strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def _latest_agent_entry(data: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """Return the most recent agent entry across the plan/build stages, or ``None``.

    Entries are concatenated in stage then append order (plan before build,
    matching run order) and stably sorted by ``timestamp``, so the last element
    is the latest — with append order breaking equal-timestamp ties in favour
    of the later write.
    """
    entries: list[dict[str, Any]] = []
    for stage in _AGENT_STAGES:
        entries.extend(data.get(stage, []))
    if not entries:
        return None
    entries.sort(key=lambda e: str(e.get("timestamp") or ""))
    return entries[-1]


def _node_row(workspace: Path, node_id: str, state: str) -> str:
    """Render one ``status.md`` node row: id, state, latest decision + summary."""
    data = read_progress(_node_progress_path(workspace, node_id))
    latest = _latest_agent_entry(data)
    if latest is None:
        detail = "(no entries)"
    else:
        summary = _one_line(latest.get("summary", ""))
        decision = latest.get("decision")
        if decision:
            detail = f"{decision} — {summary}" if summary else str(decision)
        else:
            detail = summary or "(no summary)"
    return f"- `{node_id}` · {state} · {detail}"


def render_global_status(
    *,
    workspace: Path,
    node_states: Mapping[str, str],
    node_ids: Sequence[str],
) -> None:
    """Overwrite ``workspace/status.md`` with the whole-DAG rollup + per-node rows.

    The orchestrator is the ONLY writer of this file. The header is a rollup —
    total node count plus a count per state — and the body has one row per node
    (id, state, and the node's latest decision + one-line summary read from its
    private ``nodes/<slug>/progress.yaml``). A node with empty/missing private
    progress yields a clean "no entries" row rather than crashing. The write
    fully overwrites, so re-rendering is idempotent.

    Args:
        workspace: The shared workspace root (holds ``status.md`` and ``nodes/``).
        node_states: Final node id → state map from the scheduler; values may be
            :class:`~agent_flow.orchestration.NodeState` enums or plain strings.
        node_ids: Every node id in the graph, in the order rows should appear.
    """
    states = {node_id: _state_str(node_states.get(node_id)) for node_id in node_ids}
    counts = Counter(states.values())

    ordered_states = list(_STATE_DISPLAY_ORDER)
    ordered_states += sorted(s for s in counts if s not in _STATE_DISPLAY_ORDER)

    lines = [
        "# Global run status",
        "",
        (
            "Derived, orchestrator-owned rollup of the concurrent Execution "
            "Graph. Regenerated from each node's private "
            "`nodes/<slug>/progress.yaml`; do not edit by hand."
        ),
        "",
        f"Total nodes: {len(node_ids)}",
        "",
        "## Rollup by state",
        "",
    ]
    lines += [f"- {state}: {counts.get(state, 0)}" for state in ordered_states]
    lines += ["", "## Nodes", ""]
    if node_ids:
        lines += [_node_row(workspace, node_id, states[node_id]) for node_id in node_ids]
    else:
        lines.append("- (no nodes)")
    lines.append("")

    write_status_text(workspace / GLOBAL_STATUS_FILENAME, "\n".join(lines))


def render_global_progress(*, workspace: Path, node_ids: Sequence[str]) -> None:
    """Rewrite ``workspace/progress.yaml`` with a node-tagged aggregated timeline.

    Reads every node's private ``nodes/<slug>/progress.yaml`` and merges their
    entries into a single top-level ``timeline`` list, stamping each entry with
    its ``node`` id (as the first key) and ordering the whole list by
    ``(timestamp, node)``. Nodes whose private progress is empty or missing
    simply contribute nothing — never a crash.

    The plan phase and ``--feedback`` write ``plan_stage`` / ``human_feedback``
    into this SAME shared file BEFORE the concurrent build runs, so the write
    PRESERVES those non-node sections rather than overwriting them: any
    non-empty ``plan_stage`` / ``human_feedback`` already on disk is carried
    through verbatim alongside the freshly aggregated ``timeline``.
    ``build_stage`` is intentionally dropped — per-node build entries now live
    in ``timeline``. The orchestrator is the ONLY writer, and because the
    preserved sections round-trip unchanged, re-rendering stays idempotent.

    Args:
        workspace: The shared workspace root (holds ``progress.yaml`` and
            ``nodes/``).
        node_ids: Every node id whose private progress should be aggregated.
    """
    timeline: list[dict[str, Any]] = []
    for node_id in node_ids:
        data = read_progress(_node_progress_path(workspace, node_id))
        for stage in _ALL_STAGES:
            for entry in data.get(stage, []):
                timeline.append({"node": node_id, **entry})
    timeline.sort(key=lambda e: (str(e.get("timestamp") or ""), str(e["node"])))

    # Preserve the plan-phase / human-feedback audit trail already in the shared
    # file; overwriting with only the node timeline would silently discard it.
    existing = read_progress(workspace / GLOBAL_PROGRESS_FILENAME)
    output: dict[str, Any] = {}
    if existing[PLAN_STAGE]:
        output[PLAN_STAGE] = existing[PLAN_STAGE]
    if existing[HUMAN_FEEDBACK]:
        output[HUMAN_FEEDBACK] = existing[HUMAN_FEEDBACK]
    output["timeline"] = timeline

    (workspace / GLOBAL_PROGRESS_FILENAME).write_text(
        yaml.safe_dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
