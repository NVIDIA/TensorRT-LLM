# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight's agent probe: one tiny turn on every agent the run's roles resolve to.

An in-process turn that cannot work can still look like one that did, so each
distinct agent configuration the roles resolve to
(:func:`distinct_agents`) is asked for one command through its Bash tool:
write a nonce, the nonce's SHA-256 and the checkout's ``HEAD`` into a file in
the workspace. The file is judged, never the model's words
(:func:`read_probe_file`): a model can say anything, and cannot write a hash it
never computed. A session its site's policy would run read-only fails it too,
whatever it wrote: every file a role writes would wait on an approval. A
failure carries the turn's own error text, the CLI's stderr and what to do
about it (:func:`hint_for`).

A probe turn is an ordinary in-process turn: bounded, and leaving its prompt,
activity and answer under ``gates/agent-probe/<n>/`` beside the file it wrote.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import secrets
import shlex
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig
from agent_flow.agent_runtime import AgentConfig
from agent_flow.backends.claude_code import CLI_PATH_ENV, cli_stderr_to, launched_cli

from ...agents.roles import CODEX_SESSION_ENV, ROLES, resolve_role_agents
from ...agents.turns import InProcessTurnExecutor, TurnRequest

#: Where the probe's turns and files live, under the workspace.
AGENT_PROBE_DIR = Path("gates") / "agent-probe"
#: Where the last pass is recorded, under the workspace.
AGENT_PROBE_RECORD = AGENT_PROBE_DIR / "pass.json"
#: How long a recorded pass spares a resumed run the probe, in seconds.
AGENT_PROBE_FRESH_S = 12 * 3600.0
#: How long one probe turn may take. A CLI that cannot start fails in seconds.
AGENT_PROBE_TIMEOUT_S = 600.0
#: Set to skip the probe, for when the probe itself is what is broken.
AGENT_PROBE_SKIP_ENV = "MV2FLOW_SKIP_AGENT_PROBE"
#: The role name a probe turn and its files carry.
PROBE_ROLE = "agent_probe"
#: Set in a process that runs inside another agent's session and its sandbox.
NESTED_SESSION_ENV: tuple[str, ...] = ("SANDBOX_RUNTIME", "CLAUDECODE", CODEX_SESSION_ENV)

PROBE_SYSTEM_PROMPT = (
    "You are a preflight probe for an automated workflow. Do exactly what the user "
    "message asks, with the tool it names, and nothing else."
)

_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_EXCERPT_CHARS = 1500

LayerFactory = Callable[[AgentConfig, Path], Any]


@dataclass(frozen=True)
class ProbedAgent:
    """One distinct agent configuration, and the roles that resolve to it."""

    agent: AgentConfig
    roles: tuple[str, ...]

    @property
    def label(self) -> str:
        effort = self.agent.reasoning_effort or "default effort"
        servers = self.agent.extra_mcp_servers or {}
        extra = f", MCP {', '.join(sorted(servers))}" if servers else ""
        model = self.agent.model or "the model it is set to"
        return f"{self.agent.backend} {model} ({effort}{extra})"

    def as_dict(self) -> dict:
        return {**dataclasses.asdict(self.agent), "roles": list(self.roles)}


@dataclass
class ProbeOutcome:
    """What one agent's probe turn did."""

    probed: ProbedAgent
    #: What is wrong; empty when the turn proved itself.
    problem: str = ""
    #: The turn's answer, or the error it raised.
    said: str = ""
    #: What the CLI wrote to stderr during the turn.
    stderr: list[str] = field(default_factory=list)
    #: Where the turn's prompt, activity and answer are.
    files: str = ""
    #: The sandbox and approval policy the session ran under, when its backend says.
    settings: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem


def distinct_agents(
    task_data: Mapping[str, Any],
    roles: Sequence[str] = ROLES,
    *,
    resolved: Mapping[str, AgentConfig] | None = None,
) -> list[ProbedAgent]:
    """Group the same pinned role configurations used by the workflow's turns."""
    agents = resolve_role_agents(task_data) if resolved is None else resolved
    groups: dict[str, tuple[AgentConfig, list[str]]] = {}
    for role in roles:
        agent = agents[role]
        key = json.dumps(dataclasses.asdict(agent), sort_keys=True, default=str)
        groups.setdefault(key, (agent, []))[1].append(role)
    return [ProbedAgent(agent, tuple(names)) for agent, names in groups.values()]


def probe_inputs(
    task_data: Mapping[str, Any],
    agents: Sequence[ProbedAgent],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """What a probe pass stands for, one digest per part.

    - ``agents``: each distinct configuration, and the roles on it;
    - ``cli``: the ``claude`` binary a layer launches and its version, when an
      agent is on Claude Code (``$AGENT_FLOW_CLAUDE_CLI`` moves it);
    - ``process``: this host, whether this process runs inside another
      agent's session, and the home and configuration directory the CLI
      authenticates from;
    - ``checkout``: the checkout whose ``HEAD`` the probe reads, by path.
    """
    env = os.environ if environ is None else environ
    claude = any(probed.agent.backend == "claude-code" for probed in agents)
    parts: dict[str, Any] = {
        "agents": [probed.as_dict() for probed in agents],
        "cli": list(launched_cli()) if claude else None,
        "process": {
            "host": socket.gethostname(),
            "nested": {name: env.get(name) for name in NESTED_SESSION_ENV},
            "home": env.get("HOME"),
            "config_dir": env.get("CLAUDE_CONFIG_DIR"),
        },
        "checkout": task_data.get("trtllm_repo_path"),
    }
    return part_digests(parts)


def part_digests(parts: Mapping[str, Any]) -> dict[str, str]:
    """One short digest per part of what a recorded pass stands for (this probe's, the substrate gate's)."""
    return {
        name: hashlib.sha1(
            json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        for name, value in parts.items()
    }


def probe_command(nonce: str, out: Path, repo: str) -> str:
    """The one command a probe turn runs: the nonce, its SHA-256 and ``HEAD``, errors included."""
    quoted = shlex.quote(nonce)
    return (
        f"{{ printf '%s\\n' {quoted}; printf '%s' {quoted} | sha256sum; "
        f"git -C {shlex.quote(repo)} rev-parse HEAD; }} > {shlex.quote(str(out))} 2>&1"
    )


def probe_prompt(command: str) -> str:
    return (
        "This turn checks one thing: that your Bash tool runs commands on this host.\n\n"
        "Run exactly this command with your Bash tool, once:\n\n"
        f"    {command}\n\n"
        "Do not create or edit that file any other way, and do not repair or retry anything "
        "if the command fails. Then reply with the command's exit status and any output or "
        "error text, verbatim."
    )


def read_probe_file(out: Path, nonce: str, head: str, repo: str) -> str:
    """What is wrong with the file a probe turn was asked to write; ``""`` when it proves the turn.

    It proves the turn when its lines are the nonce, the nonce's SHA-256 as
    ``sha256sum`` prints it, and the checkout's ``HEAD``: ``head`` when
    preflight could read it itself, an object id otherwise.
    """
    try:
        text = out.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"the turn ended without writing {out}: its Bash tool did not run the command"
    except OSError as exc:
        return f"{out} cannot be read: {exc}"
    lines = [line.strip() for line in text.splitlines()]
    shown = _excerpt(text) or "(empty)"
    if not lines or lines[0] != nonce:
        return (
            f"{out} does not start with this probe's nonce, so the command did not write it: "
            f"{shown}"
        )
    digest = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    if len(lines) < 2 or lines[1].split()[:1] != [digest]:
        return (
            f"{out} holds the nonce but not its SHA-256 as the shell computes it, so no shell "
            f"wrote it: {shown}"
        )
    found = lines[2] if len(lines) > 2 else ""
    proven = found == head if head else bool(_OBJECT_ID.match(found))
    if not proven:
        said = "; ".join(lines[2:]) or "nothing"
        expected = f" (preflight reads {head})" if head else ""
        return (
            f"the turn's shell ran, but `git -C {repo} rev-parse HEAD` printed {said}{expected}: "
            "the turn cannot read the checkout it is to change"
        )
    return ""


_CLI_REFUSED = re.compile(
    r"minimum (?:required )?version|older than the minimum|no longer supported|"
    r"please (?:update|upgrade)",
    re.I,
)
_SANDBOX = re.compile(r"\bEPERM\b|srt-mux|operation not permitted|sandbox", re.I)
_READ_ONLY = re.compile(r"session ran read-only")
_AUTH = re.compile(
    r"not logged in|/login|\bEROFS\b|credentials\.lock|invalid api key|unauthori[sz]ed|"
    r"authentication|\b401\b",
    re.I,
)
_MODEL_UNAVAILABLE = re.compile(
    r"model is not supported|model[_ ]not[_ ]found|does not exist or you do not have access",
    re.I,
)


def hint_for(evidence: str, *, environ: Mapping[str, str] | None = None) -> str:
    """What to do about a failed probe whose text, stderr and file read ``evidence``."""
    env = os.environ if environ is None else environ
    hints: list[str] = []
    if _CLI_REFUSED.search(evidence):
        cli, version = launched_cli()
        which = f"{cli or 'the SDK bundled one'}" + (f", {version}" if version else "")
        hints.append(
            f"The claude CLI this process launches ({which}) is refused. Set {CLI_PATH_ENV} to "
            "a newer `claude` (`which claude` and `claude --version` in a shell where it works), "
            "or put one first on PATH, and start again."
        )
    if _READ_ONLY.search(evidence):
        hints.append(
            "Codex would run the roles read-only: the site's Codex requirements "
            "(`/etc/codex/requirements.toml`, MDM or the cloud) allow no writable sandbox. Have "
            "`workspace-write` allowed, or route the roles to Claude Code "
            "(`agents.defaults.backend: claude-code`)."
        )
    elif _SANDBOX.search(evidence):
        nested = [name for name in NESTED_SESSION_ENV if env.get(name)]
        verb = "is" if len(nested) == 1 else "are"
        inside = (
            f" This process runs inside one: {' and '.join(nested)} {verb} set." if nested else ""
        )
        if env.get(CODEX_SESSION_ENV):
            hints.append(
                "The turn ran into a sandbox, and this process runs inside a Codex session, "
                "whose sandbox its commands inherit. Start the orchestrator outside it: from a "
                "session with full access (`codex --dangerously-bypass-approvals-and-sandbox`), "
                "or, where the site's Codex requirements forbid that, as a command you approve "
                "to run outside the sandbox."
            )
        else:
            hints.append(
                "The turn's Bash tool could not start its own sandbox. Launch the orchestrator "
                "from a plain terminal or a tmux session on the login node, not from inside "
                f"another agent's session.{inside}"
            )
    if _AUTH.search(evidence):
        hints.append(
            "The CLI cannot authenticate from this process. Log in with `claude` (Claude Code) "
            "or `codex login` (Codex) in the shell that launches the run (the CLIs read HOME, "
            "CLAUDE_CONFIG_DIR and CODEX_HOME from it), or launch from one that is logged in."
        )
    if _MODEL_UNAVAILABLE.search(evidence):
        hints.append(
            "This login does not offer the model the turn asked for. Set "
            "`agents.defaults.model` or the role's `model` to an available model, "
            "or launch from a session using that model."
        )
    hints.append(f"If the probe itself is what is broken, {AGENT_PROBE_SKIP_ENV}=1 skips it.")
    return "\n".join(f"- {hint}" for hint in hints)


def make_probe_layer(agent: AgentConfig, workspace: Path) -> AgentLayer:
    """The layer a role on ``agent`` runs on, without the role."""
    return AgentLayer(
        AgentLayerConfig(
            name=PROBE_ROLE,
            system_prompt=PROBE_SYSTEM_PROMPT,
            backend=BackendConfig(
                kind=agent.backend,
                model=agent.model,
                reasoning_effort=agent.reasoning_effort,
                extra_mcp_servers=agent.extra_mcp_servers,
                cwd=workspace,
            ),
            print_activity=False,
        )
    )


def _probe_one(
    probed: ProbedAgent,
    directory: Path,
    workspace: Path,
    repo: str,
    head: str,
    layer_for: LayerFactory,
    timeout_s: float,
) -> ProbeOutcome:
    out = directory / "probe.out"
    directory.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    nonce = secrets.token_hex(16)
    request = TurnRequest(
        role=PROBE_ROLE,
        what=probed.label,
        prompt=probe_prompt(probe_command(nonce, out, repo)),
        system_prompt=PROBE_SYSTEM_PROMPT,
        backend=probed.agent.backend,
        model=probed.agent.model,
        reasoning_effort=probed.agent.reasoning_effort,
    )
    stderr: list[str] = []
    layer = None
    try:
        layer = layer_for(probed.agent, workspace)
        executor = InProcessTurnExecutor(lambda _: layer, timeout_s=timeout_s, records=directory)
        with cli_stderr_to(stderr.append):
            said = executor.run(request).text
        raised = ""
    # Whatever the turn raised -- a layer that could not be built, a refused
    # CLI, a process that would not start, the probe's own bound -- is the
    # evidence this probe exists for.
    except Exception as exc:  # noqa: BLE001
        said, raised = "", f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(layer, "__exit__", None)
        if callable(close):
            close(None, None, None)
    problem = f"the turn failed: {raised}" if raised else read_probe_file(out, nonce, head, repo)
    session = _latest_session(directory)
    sandbox, approval = session.get("sandbox"), session.get("approval_policy")
    settings = ", ".join(
        f"{name} {value}"
        for name, value in (("sandbox", sandbox), ("approvals", approval))
        if value
    )
    if not problem and sandbox == "read-only":
        problem = (
            f"the session ran read-only ({settings}): every file a role writes would wait on an "
            "approval"
        )
    return ProbeOutcome(
        probed, problem, said=said, stderr=stderr, files=_latest_turn(directory), settings=settings
    )


def _latest_id(directory: Path) -> str:
    ids = [path.name[: -len(".request.json")] for path in directory.glob("*.request.json")]
    ids.sort(key=lambda name: int(name.split("-", 1)[0]) if name[:1].isdigit() else 0)
    return ids[-1] if ids else ""


def _latest_turn(directory: Path) -> str:
    """The newest turn's files in ``directory``, as a pattern an operator can list."""
    latest = _latest_id(directory)
    return str(directory / f"{latest}.*") if latest else str(directory)


def _latest_session(directory: Path) -> dict:
    """The session the newest turn in ``directory`` ran in, as its record kept it; ``{}`` if none."""
    latest = _latest_id(directory)
    try:
        sessions = json.loads((directory / f"{latest}.done").read_text())["sessions"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return sessions[-1] if sessions and isinstance(sessions[-1], dict) else {}


def probe_agents(
    agents: Sequence[ProbedAgent],
    workspace: Path | str,
    *,
    repo: str,
    head: str = "",
    layer_for: LayerFactory | None = None,
    timeout_s: float = AGENT_PROBE_TIMEOUT_S,
) -> list[ProbeOutcome]:
    """Probe every agent at once; one outcome each, in order.

    The ``n``-th agent's turn and file are under ``gates/agent-probe/<n>/``.
    ``head`` is the checkout's ``HEAD`` as preflight reads it, ``""`` when it
    could not. ``layer_for`` builds the layer for one agent
    (:func:`make_probe_layer` by default).
    """
    workspace = Path(workspace)
    make = layer_for or make_probe_layer
    # The turn runs in the workspace, so the paths its command names are absolute.
    repo = str(Path(repo).absolute()) if repo else repo
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        futures = [
            pool.submit(
                _probe_one,
                probed,
                (workspace / AGENT_PROBE_DIR / str(index)).absolute(),
                workspace,
                repo,
                head,
                make,
                timeout_s,
            )
            for index, probed in enumerate(agents, 1)
        ]
        return [future.result() for future in futures]


def _excerpt(text: str) -> str:
    text = text.strip()
    if len(text) > _EXCERPT_CHARS:
        text = f"{text[:_EXCERPT_CHARS]} ... ({len(text) - _EXCERPT_CHARS} more characters)"
    return text.replace("\n", "\n    ")


def format_failures(
    outcomes: Sequence[ProbeOutcome], *, environ: Mapping[str, str] | None = None
) -> str:
    """Each failed probe with what its turn said, then what to do."""
    failed = [outcome for outcome in outcomes if not outcome.ok]
    blocks, evidence = [], []
    for outcome in failed:
        roles = ", ".join(outcome.probed.roles)
        lines = [f"{outcome.probed.label}, for {roles}: {_excerpt(outcome.problem)}"]
        if outcome.said:
            lines.append(f"  the turn said: {_excerpt(outcome.said)}")
        if outcome.stderr:
            tail = "\n".join(outcome.stderr[-20:])
            lines.append(f"  the CLI's stderr: {_excerpt(tail)}")
        lines.append(f"  the turn's prompt, activity and answer: {outcome.files}")
        blocks.append("\n".join(lines))
        evidence += [outcome.problem, outcome.said, *outcome.stderr]
    blocks.append(hint_for("\n".join(evidence), environ=environ))
    return "\n\n".join(blocks)


__all__ = [
    "AGENT_PROBE_DIR",
    "AGENT_PROBE_FRESH_S",
    "AGENT_PROBE_RECORD",
    "AGENT_PROBE_SKIP_ENV",
    "AGENT_PROBE_TIMEOUT_S",
    "NESTED_SESSION_ENV",
    "PROBE_ROLE",
    "ProbeOutcome",
    "ProbedAgent",
    "distinct_agents",
    "format_failures",
    "hint_for",
    "make_probe_layer",
    "part_digests",
    "probe_agents",
    "probe_command",
    "probe_inputs",
    "probe_prompt",
    "read_probe_file",
]
