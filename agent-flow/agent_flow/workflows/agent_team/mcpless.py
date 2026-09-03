"""No-in-process-MCP mode support for the agent-team workflow.

Every role in the workflow normally gets a set of in-process (SDK) MCP
tools — ``append_*_progress``, ``read_latest_progress``,
``read_human_feedback``, ``update_status``, ``read_status``, ``ask_human``
— which the backend registers as a dynamically configured MCP server
(``agent-tools`` on the claude-code backend, ``dynamicTools`` on the codex
backend). Some environments forbid that: an enterprise-managed MCP config
makes Claude Code refuse to start when the SDK configures any MCP server
of its own, and a hardened runner may block dynamic tool registration
outright.

``--no-mcp-tools`` switches the workflow into the mode this module
supports: every role runs with ``tools=None`` (only the backend's own
built-in tools remain — reading, editing, and running commands are not
MCP servers, so they are unaffected). The progress/status coordination the
MCP tools used to provide is re-plumbed here:

- **reads** — the orchestrator gathers the same slices the ``read_*`` tools
  returned and inlines them into the prompt (:func:`gather_context`);
- **writes** — the agent writes a small fixed-schema per-turn handoff file
  (:func:`handoff_path`) with whatever file-writing tool its backend has,
  which the orchestrator parses (:func:`parse_handoff`) and records into
  ``progress.yaml`` via the existing writers.

:func:`build_recording_preamble` produces the authoritative instruction
block the orchestrator prepends to each turn's prompt so the agent follows
the file-based protocol instead of calling the (now unavailable) MCP tools.
It deliberately names no backend-specific tool as mandatory: the workflow
mixes claude-code roles (which have ``Write``) with codex roles (which
have ``apply_patch`` / shell), and both must be able to satisfy it.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .progress import BUILD_STAGE, HUMAN_FEEDBACK, PLAN_STAGE, find_entries, read_progress
from .status import read_status_text

# --------------------------------------------------------------------------
# Handoff schema + parsing
# --------------------------------------------------------------------------


class HandoffError(ValueError):
    """Raised when a per-turn handoff file is missing or fails validation."""


@dataclass(frozen=True)
class HandoffSchema:
    """The fields a role must write into its handoff file.

    Mirrors the JSON schema of the role's old ``append_*_progress`` MCP tool.
    """

    required: tuple[str, ...]
    decision_values: tuple[str, ...] | None
    needs_score: bool


# ``HUMAN_APPROVED`` is intentionally absent from plan_drafter's decisions:
# it required ``ask_human``, which is disabled in no-MCP mode.
HANDOFF_SCHEMAS: dict[str, HandoffSchema] = {
    "plan_drafter": HandoffSchema(
        ("summary", "decision"), ("DRAFT_READY", "POLISHING", "DONE"), False
    ),
    "plan_reviewer": HandoffSchema(("summary", "decision"), ("APPROVE", "REJECT"), False),
    "coder": HandoffSchema(("summary",), None, False),
    "reviewer": HandoffSchema(("summary", "decision"), ("APPROVE", "REJECT"), False),
    "qa": HandoffSchema(("summary", "decision", "weighted_score"), ("APPROVE", "REJECT"), True),
}

# The MCP tools each role used to have, named in the preamble so the model
# understands which later instructions the file-based protocol replaces.
_OLD_TOOLS: dict[str, tuple[str, ...]] = {
    "plan_drafter": (
        "append_plan_drafter_progress",
        "read_latest_progress",
        "read_latest_build_progress",
        "read_human_feedback",
    ),
    "plan_reviewer": (
        "append_plan_reviewer_progress",
        "read_latest_progress",
        "read_human_feedback",
    ),
    "coder": (
        "append_coder_progress",
        "read_latest_progress",
        "read_human_feedback",
        "update_status",
        "read_status",
    ),
    "reviewer": (
        "append_reviewer_progress",
        "read_latest_progress",
        "read_human_feedback",
        "update_status",
        "read_status",
    ),
    "qa": ("append_qa_progress", "read_human_feedback"),
}


def parse_handoff(role: str, text: str) -> dict[str, Any]:
    """Parse and validate a role's handoff YAML into a progress entry payload.

    Returns a dict with ``summary`` and, per the role's schema, ``decision``
    and/or ``weighted_score``. Raises :class:`HandoffError` on any problem —
    invalid YAML, non-mapping, missing/empty field, bad decision enum, or an
    out-of-range score. ``role`` must be a key of :data:`HANDOFF_SCHEMAS`.
    """
    schema = HANDOFF_SCHEMAS[role]
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HandoffError(f"{role} handoff is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HandoffError(f"{role} handoff must be a YAML mapping, got {type(data).__name__}")

    for field in schema.required:
        if field not in data or data[field] is None or data[field] == "":
            raise HandoffError(f"{role} handoff missing required field {field!r}")

    entry: dict[str, Any] = {}
    summary = str(data["summary"]).strip()
    if not summary:
        raise HandoffError(f"{role} handoff has an empty summary")
    entry["summary"] = summary

    if schema.decision_values is not None:
        decision = str(data["decision"]).strip()
        if decision not in schema.decision_values:
            raise HandoffError(
                f"{role} handoff decision {decision!r} is not one of {list(schema.decision_values)}"
            )
        entry["decision"] = decision

    if schema.needs_score:
        raw = data["weighted_score"]
        try:
            score = float(raw)
        except (TypeError, ValueError) as exc:
            raise HandoffError(f"{role} handoff weighted_score is not a number: {raw!r}") from exc
        if not 0.0 <= score <= 10.0:
            raise HandoffError(f"{role} handoff weighted_score {score} is outside [0, 10]")
        entry["weighted_score"] = score

    return entry


# --------------------------------------------------------------------------
# Context injection + recording preamble
# --------------------------------------------------------------------------


def handoff_path(turn_dir: Path, role: str) -> Path:
    """Return the per-turn handoff file path for ``role`` under ``turn_dir``."""
    return turn_dir / f"{role}.yaml"


def _yaml_block(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _render_sections(sections: list[tuple[str, str]]) -> str:
    parts = []
    for title, body in sections:
        body = body.strip() or "(none)"
        parts.append(f"### {title}\n{body}")
    return "\n\n".join(parts)


def gather_context(
    role: str,
    *,
    progress_path: Path,
    status_path: Path,
    replan: bool = False,
    feedback_triggered: bool = False,
) -> str:
    """Assemble the inline context that replaces a role's ``read_*`` tools.

    Mirrors, per role, the slices the old read tools returned:
    recent progress entries for the role's phase, the ``human_feedback``
    list, and (for coder/reviewer) the current ``status.md``.
    """

    def _entries(entries: list[dict[str, Any]]) -> str:
        return _yaml_block(entries) if entries else "(no entries yet)"

    def _feedback() -> str:
        fb = read_progress(progress_path)[HUMAN_FEEDBACK]
        return _yaml_block(fb) if fb else "(no human feedback)"

    def _status() -> str:
        return read_status_text(status_path) or "(status.md is empty)"

    sections: list[tuple[str, str]] = []
    if role == "coder":
        sections.append(
            (
                "Recent build-stage progress (last 2 iterations — Reviewer/QA "
                "REJECT feedback to address)",
                _entries(find_entries(progress_path, stage=BUILD_STAGE, last_iterations=2)),
            )
        )
        sections.append(("Human feedback (from --feedback)", _feedback()))
        sections.append(("Current status.md (rolling scratchpad)", _status()))
    elif role == "reviewer":
        sections.append(
            (
                "Coder's latest progress entry",
                _entries(
                    find_entries(progress_path, stage=BUILD_STAGE, agent="coder", last_iterations=1)
                ),
            )
        )
        sections.append(("Human feedback (from --feedback)", _feedback()))
        sections.append(("Current status.md (rolling scratchpad)", _status()))
    elif role == "plan_reviewer":
        sections.append(
            (
                "PlanDrafter's latest progress entry",
                _entries(
                    find_entries(
                        progress_path, stage=PLAN_STAGE, agent="plan_drafter", last_iterations=1
                    )
                ),
            )
        )
        if feedback_triggered:
            sections.append(("Human feedback (from --feedback)", _feedback()))
    elif role == "plan_drafter":
        if replan:
            sections.append(
                (
                    "Latest build-stage progress (coder/reviewer/qa findings to respond to)",
                    _entries(find_entries(progress_path, stage=BUILD_STAGE, last_iterations=1)),
                )
            )
            sections.append(("Human feedback (from --feedback)", _feedback()))
        else:
            sections.append(
                (
                    "Recent plan-stage progress",
                    _entries(find_entries(progress_path, stage=PLAN_STAGE, last_iterations=2)),
                )
            )
    elif role == "qa":
        sections.append(("Human feedback (from --feedback)", _feedback()))
    else:
        raise ValueError(f"unknown role: {role!r}")

    return _render_sections(sections)


# Named without committing to one backend's tool inventory: claude-code
# roles have ``Write``, codex roles have ``apply_patch`` / shell, and
# ``--no-mcp-tools`` disables tools for both at once.
_WRITE_TOOL_HINT = (
    "using whatever file-writing tool your environment provides (`Write` on "
    "Claude Code; `apply_patch` or a shell redirection on Codex)"
)


def build_recording_preamble(
    role: str,
    handoff: Path,
    status_path: Path,
    context: str,
) -> str:
    """Build the authoritative protocol block prepended to a turn's prompt.

    Declares the role's MCP tools unavailable, gives the file-based
    equivalent (writing ``handoff`` with the role's fixed YAML keys, plus a
    ``status.md`` overwrite for coder/reviewer), and inlines ``context``.
    """
    schema = HANDOFF_SCHEMAS[role]

    spec_lines = ["summary: |", "  <a short human-readable summary>"]
    if schema.decision_values:
        spec_lines.append(f"decision: <one of: {', '.join(schema.decision_values)}>")
    if schema.needs_score:
        spec_lines.append("weighted_score: <a number from 0 to 10>")
    yaml_spec = textwrap.indent("\n".join(spec_lines), "       ")

    tool_names = _OLD_TOOLS.get(role, ())
    if tool_names:
        tools_sentence = (
            "The MCP tools this workflow normally provides ("
            + ", ".join(f"`{t}`" for t in tool_names)
            + ") are NOT available in this run."
        )
    else:
        tools_sentence = "In-process MCP tools are NOT available in this run."

    status_clause = ""
    if role in ("coder", "reviewer"):
        status_clause = (
            "\n3. To update the rolling status (replacing any `update_status` call), "
            f"OVERWRITE `{status_path}` — with the same file-writing tool — with a "
            "fresh, self-contained status snapshot (current status, execution path, "
            "what was tried, what worked / didn't, pointers for the next step)."
        )

    return (
        "=== RUN MODE: MCP tools disabled (--no-mcp-tools) ===\n"
        f"{tools_sentence} Wherever the instructions below tell you to CALL one of those "
        "tools, follow this protocol instead:\n"
        "1. Any prior progress / human-feedback / status you would have fetched with a "
        "`read_*` tool is already provided inline under CONTEXT below — do not try to read "
        "progress.yaml or status.md to get it.\n"
        "2. To RECORD your progress entry (replacing any `append_*_progress` call), write "
        f"the following file as the LAST action of your turn, {_WRITE_TOOL_HINT}:\n"
        f"       {handoff}\n"
        "   containing exactly these YAML keys and nothing else:\n"
        f"{yaml_spec}"
        f"{status_clause}\n"
        "\n=== CONTEXT ===\n"
        f"{context}\n"
        "=== END CONTEXT ===\n"
    )
