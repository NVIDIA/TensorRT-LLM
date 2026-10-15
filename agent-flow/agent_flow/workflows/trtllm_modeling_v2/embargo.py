# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``source_embargo``, enforced: a tool-call hook, not only a sentence in a prompt.

``task.yaml`` names files and trees no agent may read — an implementation of
this exact checkpoint that already exists, here or in a sibling engine's
checkout. Every role prompt says so (``prompts._common.source_embargo_block``),
and a prompt is a request. This module makes it a mechanism for every role
the orchestrator spawns in process, through the agent backend's native
hooks (``BackendConfig.hooks``):

**PreToolUse — refuse the read.**

- ``Read`` / ``Edit`` / ``MultiEdit`` / ``Write`` / ``NotebookEdit`` on an
  embargoed file, or anywhere inside an embargoed tree (symlinks resolved).
- ``Grep`` whose search root is an embargoed file or lies inside an
  embargoed tree.
- ``Bash`` whose command names an embargoed **file** (absolute,
  repo-relative or by its file name, or through a glob that matches it), or
  names an embargoed **tree** together with a reading or searching program
  (``cat``, ``grep``, ``sed``, ``python -c`` ...). Naming a tree to *run*
  it — starting a sibling engine as the accuracy oracle — stays allowed:
  running the other implementation is expected; reading it is not.

**PostToolUse — redact what a broad search swept in.** A ``Grep`` or a
``grep -r`` over a parent directory is legitimate and can still return lines
from an embargoed file. Every output line (and every file-list element) that
names an embargoed path is removed before the model sees it, the output
keeps its shape, and the agent is told how many lines were withheld.

**Every refusal and every redaction is logged**, one JSON object per line,
to ``<workspace>/embargo/attempts.jsonl``: the role, the tool, the matched
entry, and what was asked.

What it cannot see: a command that reads an embargoed file without naming
it or any parent search root that output lines would name (``rg -h``, a
script that opens a computed path). The prompt rule still covers those, and
the log records everything the hook did see.

Where no hook can run, the embargo is an instruction instead, appended to the
role's system prompt by :func:`embargo_instruction`: a role on a backend
that cannot run this process's SDK callbacks (Codex refuses per-client
hooks; see :func:`source_embargo_enforcement`), and a turn delivered over the
external transport, which runs in a session this process does not own
(:func:`embargo_transport_note`).

Entries are repo-relative, absolute (a tree outside the repository), or
``package:<path>`` — relative to this workflow package wherever it is
installed, for the package's own worked examples.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EMBARGO_DIRNAME = "embargo"
ATTEMPTS_FILENAME = "attempts.jsonl"

#: Tools whose input names one file.
FILE_TOOLS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "MultiEdit": ("file_path",),
    "Write": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "NotebookRead": ("notebook_path",),
}
#: Tools whose output may carry another file's lines.
REDACTED_TOOLS: tuple[str, ...] = ("Grep", "Bash")

#: Programs that read, search or copy what they are pointed at. Naming an
#: embargoed *tree* is only refused alongside one of these (or an inline
#: interpreter, below); naming an embargoed *file* is refused outright.
READ_PROGRAMS: frozenset[str] = frozenset(
    {
        "cat", "tac", "nl", "less", "more", "most", "head", "tail", "sed", "awk", "gawk",
        "grep", "egrep", "fgrep", "rg", "ag", "ack", "git", "xargs", "cut", "sort", "uniq",
        "diff", "cmp", "comm", "strings", "xxd", "od", "hexdump", "bat", "view", "vi", "vim",
        "nvim", "nano", "emacs", "ed", "cp", "rsync", "scp", "tar", "zip", "jq", "yq",
        "base64", "dd", "paste", "fold", "fmt", "pr", "ctags", "pygmentize",
    }
)  # fmt: skip
#: Interpreters that read a path only when handed inline code. ``python -m
#: vllm...`` from inside an embargoed checkout runs it, which is expected.
INLINE_INTERPRETERS: frozenset[str] = frozenset({"python", "python3", "perl", "ruby", "node"})

_GLOB_CHARS = set("*?[")

#: An entry spelled ``package:<path>`` is relative to this workflow package,
#: wherever it is installed — e.g. ``package:examples/<model>.md``, a worked
#: example a run that onboards that model must not read.
PACKAGE_PREFIX = "package:"
PACKAGE_ROOT = Path(__file__).resolve().parent

#: Backends whose sessions run this process's SDK hook callbacks. Every other
#: backend gets the embargo as an instruction in the role's system prompt.
HOOK_BACKENDS: frozenset[str] = frozenset({"claude-code"})


class EmbargoEntryError(ValueError):
    """An embargo entry that names nothing it is allowed to name."""


def package_entry_problem(entry: str) -> str:
    """Why a ``package:`` entry is unusable, or ``""``.

    It must name something strictly **inside** the workflow package:
    ``package:../perf_analyze`` resolves to a sibling workflow, and an entry
    for the package root would embargo the workflow's own skills.
    """
    root = PACKAGE_ROOT.resolve()
    target = (PACKAGE_ROOT / entry[len(PACKAGE_PREFIX) :].lstrip("/")).resolve()
    if target == root:
        return "names the whole workflow package; name a file or directory inside it"
    if root not in target.parents:
        return f"resolves to {target}, outside the workflow package {root}"
    return ""


def resolve_entry(entry: str, repo_root: str | Path | None = None) -> Path:
    """The absolute path an embargo entry names (symlinks not resolved).

    Raises :class:`EmbargoEntryError` for a ``package:`` entry that escapes
    the workflow package; the task schema reports that before a run starts.
    """
    if entry.startswith(PACKAGE_PREFIX):
        problem = package_entry_problem(entry)
        if problem:
            raise EmbargoEntryError(f"{entry!r} {problem}")
        return PACKAGE_ROOT / entry[len(PACKAGE_PREFIX) :].lstrip("/")
    raw = Path(entry)
    if raw.is_absolute() or repo_root is None:
        return raw
    return Path(repo_root) / raw


def display_entry(entry: str, repo_root: str | Path | None = None) -> str:
    """How a role prompt names an entry.

    Repo and absolute entries pass through. A ``package:`` entry keeps its
    prefix, so the prompt's embargo block can give it its own wording — it
    is the workflow's reference material, not an implementation of the
    checkpoint — followed by the path an agent would actually meet:
    repo-relative when the package sits inside the repository, else absolute.
    """
    if not entry.startswith(PACKAGE_PREFIX):
        return entry
    absolute = resolve_entry(entry, repo_root)
    shown = str(absolute)
    if repo_root is not None:
        try:
            shown = absolute.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    return f"{PACKAGE_PREFIX}{shown}"


def display_entries(task_data: Mapping[str, Any]) -> list[str]:
    """``source_embargo`` as the role prompts render it."""
    repo = task_data.get("trtllm_repo_path") or None
    return [display_entry(str(e), repo) for e in (task_data.get("source_embargo") or [])]


@dataclass(frozen=True)
class EmbargoEntry:
    """One ``source_embargo`` entry, resolved."""

    entry: str
    path: Path
    is_dir: bool
    #: Spellings that, found in a command or an output line, name this entry.
    forms: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    entry: str = ""
    reason: str = ""


def _forms(
    entry: str, absolute: Path, resolved: Path, repo: Path | None, is_dir: bool
) -> tuple[str, ...]:
    forms = {str(absolute), str(resolved)}
    if entry.startswith(PACKAGE_PREFIX):
        forms.add(entry[len(PACKAGE_PREFIX) :].strip("/"))
        try:
            forms.add(resolved.relative_to(PACKAGE_ROOT.parent.parent.parent).as_posix())
        except ValueError:
            pass
    elif not entry.startswith("/"):
        forms.add(entry.rstrip("/"))
    if repo is not None:
        for root in {repo, repo.resolve()}:
            try:
                forms.add(resolved.relative_to(root).as_posix())
            except ValueError:
                pass
    if not is_dir:
        forms.add(absolute.name)
    return tuple(sorted(f for f in forms if f and f != "."))


def _pattern(forms: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<![\w.-])(?:{alternatives})(?![\w.-])")


class SourceEmbargo:
    """The resolved embargo list, with the checks the hooks apply."""

    def __init__(
        self,
        entries: Sequence[str],
        *,
        repo_root: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else None
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()
        resolved: list[EmbargoEntry] = []
        for entry in entries:
            absolute = resolve_entry(entry, self.repo_root)
            real = absolute.resolve()
            is_dir = real.is_dir() or entry.endswith("/")
            resolved.append(
                EmbargoEntry(
                    entry=entry,
                    path=real,
                    is_dir=is_dir,
                    forms=_forms(entry, absolute, real, self.repo_root, is_dir),
                )
            )
        self.entries: tuple[EmbargoEntry, ...] = tuple(resolved)
        self._patterns = [(e, _pattern(e.forms)) for e in self.entries]

    @classmethod
    def from_task(
        cls, task_data: Mapping[str, Any], workspace: str | Path | None = None
    ) -> "SourceEmbargo | None":
        entries = [str(p) for p in (task_data.get("source_embargo") or [])]
        if not entries:
            return None
        log = Path(workspace) / EMBARGO_DIRNAME / ATTEMPTS_FILENAME if workspace else None
        return cls(entries, repo_root=task_data.get("trtllm_repo_path") or None, log_path=log)

    # ------------------------------------------------------------- matching

    def match_path(self, path: str, cwd: str | Path | None = None) -> EmbargoEntry | None:
        """The entry ``path`` falls under, symlinks resolved, or ``None``."""
        if not path:
            return None
        raw = Path(os.path.expanduser(path))
        if not raw.is_absolute():
            raw = Path(cwd or self.repo_root or Path.cwd()) / raw
        candidates = {Path(os.path.abspath(raw)), raw.resolve()}
        for entry in self.entries:
            for candidate in candidates:
                if candidate == entry.path or (entry.is_dir and entry.path in candidate.parents):
                    return entry
        return None

    def mentioned(self, text: str) -> EmbargoEntry | None:
        """The first entry ``text`` names in any of its spellings."""
        for entry, pattern in self._patterns:
            if pattern.search(text):
                return entry
        return None

    def _glob_match(self, token: str, cwd: str | Path | None) -> EmbargoEntry | None:
        """The embargoed file a shell glob would expand to, if any.

        Tried against the session's cwd and the repository root both: the
        command may ``cd`` first, and over-matching only refuses a read.
        """
        if not _GLOB_CHARS & set(token):
            return None
        if token.startswith("/"):
            patterns = [token]
        else:
            bases = {str(b) for b in (cwd, self.repo_root) if b}
            patterns = [str(Path(b) / token) for b in bases] or [token]
        for entry in self.entries:
            if any(fnmatch.fnmatch(str(entry.path), p) for p in patterns):
                return entry
        return None

    # ------------------------------------------------------------- the checks

    def check_tool(
        self, tool: str, tool_input: Mapping[str, Any], cwd: str | Path | None = None
    ) -> Verdict:
        """Whether one tool call may run."""
        if tool in FILE_TOOLS:
            for key in FILE_TOOLS[tool]:
                hit = self.match_path(str(tool_input.get(key) or ""), cwd)
                if hit:
                    return self._deny(hit, f"{tool} on an embargoed path")
            return Verdict(True)
        if tool == "Grep":
            root = str(tool_input.get("path") or "")
            hit = self.match_path(root, cwd) if root else None
            if hit:
                return self._deny(hit, "Grep inside an embargoed path")
            return Verdict(True)
        if tool == "Bash":
            return self._check_command(str(tool_input.get("command") or ""), cwd)
        return Verdict(True)

    def _check_command(self, command: str, cwd: str | Path | None) -> Verdict:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            tokens = command.split()
        programs = {Path(t).name for t in tokens if t and not t.startswith("-")}
        inline = bool(programs & INLINE_INTERPRETERS) and bool({"-c", "-e"} & set(tokens))
        reads = bool(programs & READ_PROGRAMS) or inline or "<" in command
        hit = self.mentioned(command)
        if hit is not None and (not hit.is_dir or reads):
            return self._deny(hit, "a shell command that names an embargoed path")
        for token in tokens:
            globbed = self._glob_match(token, cwd)
            if globbed is not None and reads:
                return self._deny(globbed, "a shell glob that matches an embargoed path")
        return Verdict(True)

    def _deny(self, entry: EmbargoEntry, what: str) -> Verdict:
        return Verdict(
            False,
            entry.entry,
            f"Refused by the run's source embargo: {what} (`{entry.entry}`). This path is off "
            f"limits to read for the whole run — not to open, grep, cat or copy. Running the "
            f"implementation as a black box is still expected. This attempt has been logged; "
            f"continue without it.",
        )

    # ------------------------------------------------------------- redaction

    def redact(self, value: Any) -> tuple[Any, int]:
        """``value`` with every line / element naming an embargoed path removed.

        The structure is preserved (same keys, same types), because a hook's
        replacement output must match the tool's own output schema.
        """
        if isinstance(value, str):
            lines = value.splitlines(keepends=True)
            kept = [line for line in lines if self.mentioned(line) is None]
            return "".join(kept), len(lines) - len(kept)
        if isinstance(value, list):
            out, removed = [], 0
            for item in value:
                if isinstance(item, str) and self.mentioned(item) is not None:
                    removed += 1
                    continue
                new, n = self.redact(item)
                out.append(new)
                removed += n
            return out, removed
        if isinstance(value, dict):
            out_d, removed = {}, 0
            for key, item in value.items():
                new, n = self.redact(item)
                out_d[key] = new
                removed += n
            return out_d, removed
        return value, 0

    # ------------------------------------------------------------- the log

    def log(self, **event: Any) -> None:
        if self.log_path is None:
            return
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def attempts(self) -> list[dict[str, Any]]:
        if self.log_path is None or not self.log_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line
        ]


def _summary(tool_input: Mapping[str, Any]) -> str:
    for key in ("command", "file_path", "notebook_path", "path", "pattern"):
        if tool_input.get(key):
            return str(tool_input[key])[:300]
    return json.dumps(dict(tool_input), default=str)[:300]


def embargo_hooks(embargo: SourceEmbargo, *, role: str) -> dict[str, list[Any]]:
    """``BackendConfig.hooks`` entries enforcing ``embargo`` for one role's sessions."""
    from claude_agent_sdk.types import HookMatcher

    async def pre_tool_use(input_data, _tool_use_id, _context):
        tool = str(input_data.get("tool_name") or "")
        tool_input = input_data.get("tool_input") or {}
        verdict = embargo.check_tool(tool, tool_input, input_data.get("cwd"))
        if verdict.allowed:
            return {}
        embargo.log(
            action="deny",
            role=role,
            tool=tool,
            entry=verdict.entry,
            detail=_summary(tool_input),
            session=input_data.get("session_id"),
            agent_id=input_data.get("agent_id"),
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": verdict.reason,
            }
        }

    async def post_tool_use(input_data, _tool_use_id, _context):
        tool = str(input_data.get("tool_name") or "")
        if tool not in REDACTED_TOOLS:
            return {}
        redacted, removed = embargo.redact(input_data.get("tool_response"))
        if not removed:
            return {}
        tool_input = input_data.get("tool_input") or {}
        embargo.log(
            action="redact",
            role=role,
            tool=tool,
            removed=removed,
            detail=_summary(tool_input),
            session=input_data.get("session_id"),
            agent_id=input_data.get("agent_id"),
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": redacted,
                "additionalContext": (
                    f"The run's source embargo withheld {removed} line(s) of this {tool} output "
                    f"that came from embargoed paths. Do not try to read them another way; the "
                    f"attempt is logged."
                ),
            }
        }

    pre_matcher = "|".join([*FILE_TOOLS, "Grep", "Bash"])
    return {
        "PreToolUse": [HookMatcher(matcher=pre_matcher, hooks=[pre_tool_use])],
        "PostToolUse": [HookMatcher(matcher="|".join(REDACTED_TOOLS), hooks=[post_tool_use])],
    }


def merge_hook_maps(*maps: Mapping[str, list[Any]] | None) -> dict[str, list[Any]] | None:
    """Concatenate hook maps event by event; ``None`` when there is nothing."""
    merged: dict[str, list[Any]] = {}
    for hook_map in maps:
        for event, matchers in (hook_map or {}).items():
            merged.setdefault(event, []).extend(matchers)
    return merged or None


def source_embargo_hooks(
    task_data: Mapping[str, Any], workspace: str | Path, *, role: str
) -> dict[str, list[Any]] | None:
    """The hooks for ``role``, or ``None`` when the task embargoes nothing."""
    embargo = SourceEmbargo.from_task(task_data, workspace)
    return embargo_hooks(embargo, role=role) if embargo is not None else None


def embargo_instruction(
    paths: Sequence[str], *, repo_root: str | Path | None = None, why: str
) -> str:
    """The embargo, restated for a session no hook of this process reaches.

    ``why`` says which: a turn on the external transport runs in a session
    the orchestrator does not own; a role on a backend outside
    :data:`HOOK_BACKENDS` cannot run SDK callbacks. The role prompt still
    carries the embargo; this says that nothing will refuse the read, gives
    absolute paths, and asks for any exposure to be reported, not used.
    """
    if not paths:
        return ""
    listed = "\n".join(f"- `{resolve_entry(str(p), repo_root)}`" for p in paths)
    return (
        "\n\n---\n\n"
        "## The source embargo is not enforced by a hook here — you enforce it\n\n"
        "In process, on a backend that runs them, a hook refuses every Read, Grep and shell "
        f"read of these paths and logs the attempt. {why}, so nothing will stop you:\n\n"
        + listed
        + "\n\nA directory embargoes the whole tree beneath it. Do not open, grep, cat, copy or "
        "quote any of it, and keep broad searches out of it (exclude it, or search narrower "
        "roots). If a tool output shows you content from one anyway, stop reading it, do not "
        "use it, and list each exposure under `## Embargo exposures` in your result. "
        "Running the implementation as a black box remains expected.\n"
    )


def embargo_transport_note(paths: Sequence[str], *, repo_root: str | Path | None = None) -> str:
    """:func:`embargo_instruction` for a turn on the external transport."""
    return embargo_instruction(
        paths,
        repo_root=repo_root,
        why="This turn runs on the external transport, in a session the orchestrator does not own",
    )


def source_embargo_enforcement(
    task_data: Mapping[str, Any], workspace: str | Path, *, role: str, backend: str
) -> tuple[dict[str, list[Any]] | None, str]:
    """``(hooks, system-prompt suffix)`` enforcing the task's embargo for one role.

    A backend in :data:`HOOK_BACKENDS` gets the PreToolUse/PostToolUse hooks
    and no suffix; any other backend (Codex refuses per-client callbacks) gets
    no hooks and the instruction. Both are empty when nothing is embargoed.
    """
    embargo = SourceEmbargo.from_task(task_data, workspace)
    if embargo is None:
        return None, ""
    if backend in HOOK_BACKENDS:
        return embargo_hooks(embargo, role=role), ""
    note = embargo_instruction(
        [e.entry for e in embargo.entries],
        repo_root=embargo.repo_root,
        why=f"This role runs on the `{backend}` backend, which cannot run this process's hooks",
    )
    embargo.log(action="instruct", role=role, backend=backend, detail="no hook on this backend")
    return None, note


__all__ = [
    "ATTEMPTS_FILENAME",
    "EMBARGO_DIRNAME",
    "EmbargoEntry",
    "EmbargoEntryError",
    "SourceEmbargo",
    "Verdict",
    "HOOK_BACKENDS",
    "PACKAGE_PREFIX",
    "PACKAGE_ROOT",
    "display_entries",
    "display_entry",
    "embargo_hooks",
    "embargo_instruction",
    "embargo_transport_note",
    "merge_hook_maps",
    "package_entry_problem",
    "resolve_entry",
    "source_embargo_enforcement",
    "source_embargo_hooks",
]
