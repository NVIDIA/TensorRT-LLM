# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight's agent probe: one tiny turn on every agent the run's roles resolve to.

A turn that cannot work can still look like one that did, so each distinct agent
configuration is asked for one command through its Bash tool: write a nonce, the nonce's
SHA-256 and the checkout's ``HEAD`` into a file in the workspace. The file is judged, never
the model's words: a model can say anything, and cannot write a hash it never computed. A
session that ran read-only fails too: every file a role writes would wait on an approval.
A failure carries the turn's own text, the CLI's stderr and what to do (:func:`hint_for`).
Each probe turn leaves its prompt, activity and answer under ``gates/agent-probe/<n>/``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import secrets
import shlex
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig
from agent_flow.agent_runtime import AgentConfig
from agent_flow.backends.claude_code import CLI_PATH_ENV, cli_stderr_to, launched_cli

from ...agents.roles import ROLES, resolve_role_agents
from ...agents.turns import InProcessTurnExecutor, TurnRequest

#: Where the probe's turns and files live, under the workspace.
AGENT_PROBE_DIR = Path("gates") / "agent-probe"
#: How long one probe turn may take. A CLI that cannot start fails in seconds.
AGENT_PROBE_TIMEOUT_S = 600.0
#: Set to skip the probe, for when the probe itself is what is broken.
AGENT_PROBE_SKIP_ENV = "MV2FLOW_SKIP_AGENT_PROBE"
PROBE_ROLE = "agent_probe"
#: Set in a process that runs inside another agent's session and its sandbox.
NESTED_SESSION_ENV: tuple[str, ...] = ("SANDBOX_RUNTIME", "CLAUDECODE", "CODEX_THREAD_ID")
PROBE_SYSTEM_PROMPT = (
    "You are a preflight probe for an automated workflow. Do exactly what the user "
    "message asks, with the tool it names, and nothing else."
)
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_NO_POLICY = "the workflow does not change that policy"
#: What a failed probe's evidence shows and what to do about it: of each group, the first
#: pattern found says its hint. ``{cli}`` is the CLI this process launches.
_HINTS: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        (
            r"minimum (?:required )?version|older than the minimum|no longer supported|please up",
            f"The claude CLI this process launches ({{cli}}) is refused: set {CLI_PATH_ENV} to a "
            "compatible, site-approved `claude` and start again.",
        ),
    ),
    (
        (
            r"session ran read-only",
            "The native CLI session is read-only: give it the workspace access it needs in the "
            f"CLI or its deployment environment; {_NO_POLICY}.",
        ),
        (
            r"blocked by the Bash tool|Command was blocked|expansion obfuscation|requires "
            r"approval|permission to use Bash|haven't granted it yet",
            "The CLI's permission policy refused the command, and no one approves it in an "
            "unattended turn: let the CLI the run launches run commands without approval, e.g. "
            '`{{"permissions": {{"defaultMode": "bypassPermissions"}}}}` in the settings.json of '
            "its CLAUDE_CONFIG_DIR (Claude Code refuses that mode as root unless IS_SANDBOX=1); "
            f"{_NO_POLICY}.",
        ),
        (
            r"\bEPERM\b|srt-mux|operation not permitted|sandbox",
            "The native CLI could not execute under its sandbox or permission policy: launch "
            f"from an authorized environment; {_NO_POLICY}.{{nested}}",
        ),
    ),
    (
        (
            r"not logged in|/login|\bEROFS\b|credentials\.lock|invalid api key|unauthori[sz]ed|"
            r"authentication|\b401\b",
            "The CLI cannot authenticate from this process: log in with `claude` or `codex "
            "login` in the shell that launches the run (it reads HOME, CLAUDE_CONFIG_DIR and "
            "CODEX_HOME), or launch from one that is logged in.",
        ),
    ),
    (
        (
            r"model is not supported|model[_ ]not[_ ]found|does not exist or you do not have access",
            "This login does not offer the model the turn asked for: set `agents.defaults.model` "
            "or the role's `model` to one it offers.",
        ),
    ),
)

LayerFactory = Callable[[AgentConfig, Path], Any]


@dataclass(frozen=True)
class ProbedAgent:
    """One distinct agent configuration, and the roles that resolve to it."""

    agent: AgentConfig
    roles: tuple[str, ...]

    @property
    def label(self) -> str:
        servers = self.agent.extra_mcp_servers or {}
        extra = f", MCP {', '.join(sorted(servers))}" if servers else ""
        effort = self.agent.reasoning_effort or "default effort"
        return f"{self.agent.backend} {self.agent.model or 'its own model'} ({effort}{extra})"


@dataclass
class ProbeOutcome:
    """What one agent's probe turn did."""

    probed: ProbedAgent
    #: What is wrong; empty when the turn proved itself.
    problem: str = ""
    #: The turn's answer, what the CLI wrote to stderr, and where the turn's files are.
    said: str = ""
    stderr: list[str] = field(default_factory=list)
    files: str = ""
    #: The sandbox and approval policy the session ran under, and the skills it loaded.
    settings: str = ""
    skills: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problem

    def describe(self) -> str:
        ran = f" [{self.settings}]" if self.settings else ""
        return f"{self.probed.label}{ran}: {', '.join(self.probed.roles)}"


def distinct_agents(
    task_data: Mapping[str, Any],
    roles: Sequence[str] = ROLES,
    *,
    resolved: Mapping[str, AgentConfig] | None = None,
) -> list[ProbedAgent]:
    """The distinct configurations the run's roles resolve to, each with its roles."""
    agents = resolve_role_agents(task_data) if resolved is None else resolved
    groups: dict[str, tuple[AgentConfig, list[str]]] = {}
    for role in roles:
        key = json.dumps(dataclasses.asdict(agents[role]), sort_keys=True, default=str)
        groups.setdefault(key, (agents[role], []))[1].append(role)
    return [ProbedAgent(agent, tuple(names)) for agent, names in groups.values()]


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
        f"Run exactly this command with your Bash tool, once:\n\n    {command}\n\n"
        "Do not create or edit that file any other way, and do not repair or retry anything "
        "if the command fails. Then reply with the command's exit status and any output or "
        "error text, verbatim."
    )


def _excerpt(text: str, limit: int = 1500) -> str:
    text = text.strip()
    cut = f" ... ({len(text) - limit} more characters)" if len(text) > limit else ""
    return (text[:limit] + cut).replace("\n", "\n    ")


def read_probe_file(out: Path, nonce: str, head: str, repo: str) -> str:
    """What is wrong with the file a probe turn was to write; ``""`` when it proves the turn.

    Its lines are the nonce, the nonce's SHA-256 as ``sha256sum`` prints it, and the
    checkout's ``HEAD``: ``head`` when preflight could read it, an object id otherwise.
    """
    try:
        text = out.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"the turn ended without writing {out}: its Bash tool did not run the command"
    lines = [line.strip() for line in text.splitlines()]
    shown = _excerpt(text) or "(empty)"
    if not lines or lines[0] != nonce:
        return (
            f"{out} does not start with this probe's nonce: the command did not write it: {shown}"
        )
    digest = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    if len(lines) < 2 or lines[1].split()[:1] != [digest]:
        return f"{out} holds the nonce but not its SHA-256: no shell wrote it: {shown}"
    found = lines[2] if len(lines) > 2 else ""
    if not (found == head if head else _OBJECT_ID.match(found)):
        expected = f" (preflight reads {head})" if head else ""
        return (
            f"the turn's shell ran, but `git -C {repo} rev-parse HEAD` printed "
            f"{'; '.join(lines[2:]) or 'nothing'}{expected}: the turn cannot read the checkout"
        )
    return ""


def hint_for(evidence: str, *, environ: Mapping[str, str] | None = None) -> str:
    """What to do about a failed probe whose text, stderr and file read ``evidence``."""
    env = os.environ if environ is None else environ
    hints = [
        next((hint for pattern, hint in group if re.search(pattern, evidence, re.I)), "")
        for group in _HINTS
    ]
    hints.append(f"If the probe itself is what is broken, {AGENT_PROBE_SKIP_ENV}=1 skips it.")
    nested = [name for name in NESTED_SESSION_ENV if env.get(name)]
    fills = {
        "nested": f" Parent-session markers are present: {', '.join(nested)}." if nested else ""
    }
    if any("{cli}" in hint for hint in hints):
        cli, version = launched_cli()
        fills["cli"] = (cli or "the SDK's bundled one") + (f", {version}" if version else "")
    return "\n".join(f"- {hint.format(**fills)}" for hint in hints if hint)


def make_probe_layer(agent: AgentConfig, workspace: Path) -> AgentLayer:
    """The backend configuration a role uses: without the user's plugins, its skills on."""
    backend = BackendConfig(
        kind=agent.backend,
        model=agent.model,
        reasoning_effort=agent.reasoning_effort,
        extra_mcp_servers=agent.extra_mcp_servers,
        cwd=workspace,
        plugins=False,
    )
    config = AgentLayerConfig(
        name=PROBE_ROLE, system_prompt=PROBE_SYSTEM_PROMPT, backend=backend, print_activity=False
    )
    return AgentLayer(config)


def _latest(directory: Path) -> tuple[str, dict]:
    """The newest turn's files in ``directory``, as a pattern, and the session it ran in."""
    ids = [path.name[: -len(".request.json")] for path in directory.glob("*.request.json")]
    ids.sort(key=lambda name: int(name.split("-", 1)[0]) if name[:1].isdigit() else 0)
    if not ids:
        return str(directory), {}
    try:
        sessions = json.loads((directory / f"{ids[-1]}.done").read_text())["sessions"]
        session = sessions[-1] if isinstance(sessions[-1], dict) else {}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        session = {}
    return str(directory / f"{ids[-1]}.*"), session


def _probe_one(probed, directory: Path, workspace: Path, repo: str, head: str, make, timeout_s):
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
    said, raised, layer = "", "", None
    try:
        layer = make(probed.agent, workspace)
        executor = InProcessTurnExecutor(lambda _: layer, timeout_s=timeout_s, records=directory)
        with cli_stderr_to(stderr.append):
            said = executor.run(request).text
    # Whatever the turn raised (a refused CLI, a process that would not start, the probe's
    # own bound) is the evidence this probe exists for.
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        if callable(getattr(layer, "__exit__", None)):
            layer.__exit__(None, None, None)
    problem = f"the turn failed: {raised}" if raised else read_probe_file(out, nonce, head, repo)
    files, session = _latest(directory)
    ran = (("sandbox", session.get("sandbox")), ("approvals", session.get("approval_policy")))
    settings = ", ".join(f"{name} {value}" for name, value in ran if value)
    if not problem and session.get("sandbox") == "read-only":
        problem = f"the session ran read-only ({settings}): a role's every write would wait"
    skills = session.get("skills") if isinstance(session.get("skills"), list) else ()
    return ProbeOutcome(probed, problem, said, stderr, files, settings, tuple(map(str, skills)))


def probe_agents(
    agents: Sequence[ProbedAgent],
    workspace: Path | str,
    *,
    repo: str,
    head: str = "",
    layer_for: LayerFactory | None = None,
    timeout_s: float = AGENT_PROBE_TIMEOUT_S,
) -> list[ProbeOutcome]:
    """Probe every agent at once; one outcome each, in order, under ``gates/agent-probe/<n>/``.

    ``head`` is the checkout's ``HEAD`` as preflight reads it, ``""`` when it could not.
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


def foreign_skills(outcomes: Sequence[ProbeOutcome]) -> dict[str, tuple[str, ...]]:
    """Backend kind -> the skills its probe session loaded: the user's, which no worker is told of."""
    found: dict[str, set[str]] = {}
    for outcome in outcomes:
        found.setdefault(outcome.probed.agent.backend, set()).update(outcome.skills)
    return {backend: tuple(sorted(names)) for backend, names in found.items()}


def format_failures(
    outcomes: Sequence[ProbeOutcome], *, environ: Mapping[str, str] | None = None
) -> str:
    """Each failed probe with what its turn said, then what to do."""
    blocks, evidence = [], []
    for outcome in (o for o in outcomes if not o.ok):
        probed = outcome.probed
        lines = [f"{probed.label}, for {', '.join(probed.roles)}: {_excerpt(outcome.problem)}"]
        if outcome.said:
            lines.append(f"  the turn said: {_excerpt(outcome.said)}")
        if outcome.stderr:
            lines.append(f"  the CLI's stderr: {_excerpt(chr(10).join(outcome.stderr[-20:]))}")
        lines.append(f"  the turn's prompt, activity and answer: {outcome.files}")
        blocks.append("\n".join(lines))
        evidence += [outcome.problem, outcome.said, *outcome.stderr]
    return "\n\n".join([*blocks, hint_for("\n".join(evidence), environ=environ)])
