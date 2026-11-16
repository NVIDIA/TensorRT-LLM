"""Check whether a skill is loaded by the Claude and Codex harnesses.

The check is performed by connecting a real ``AgentLayer`` session for
each backend and asking it what it loaded
(``BackendClient.list_available_skills``). Asking the harness is the only
authoritative answer: which skills resolve is the CLI's own merge of user
settings, project settings, the ``enabledPlugins`` map and the installed
marketplaces. No model call is involved — both backends answer from the
session the client already established — so a probe costs a process spawn
and no tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class AgentSkillProbe:
    """Result of probing one backend via ``AgentLayer.fetch_available_skills``.

    ``skills`` is what that backend reports as invocable. Claude Code
    answers from its command table, a superset of what the *model* can
    invoke, so a hit means "installed under this name", not "``Skill``
    will load it" — leave the agent's own load attempt as the confirming
    check.
    """

    backend_kind: str
    skills: list[str] = field(default_factory=list)
    error: str | None = None  # None on success; str message when the probe failed

    @property
    def reachable(self) -> bool:
        return self.error is None

    def has(self, skill: str) -> bool:
        """Exact-name membership.

        Note that plugin-provided skills are reported **only** in their
        fully-qualified ``<plugin>:<name>`` form, so a bare name never
        matches one. Use :meth:`resolve` unless you specifically mean an
        exact match.
        """
        return skill in self.skills

    def resolve(self, skill: str) -> str | None:
        """Return the loaded name for ``skill``, or ``None`` if absent.

        Accepts either spelling and answers with the name the harness
        actually loaded, so callers can quote it back to an agent:
        a bare ``perf-analysis`` matches a loaded
        ``trtllm-agent-toolkit:perf-analysis``, and a fully-qualified
        request matches itself. Exact matches win over suffix matches so
        an unqualified local skill is never shadowed by a plugin's.
        """
        if skill in self.skills:
            return skill
        if ":" in skill:
            return None
        suffix = f":{skill}"
        for loaded in self.skills:
            if loaded.endswith(suffix):
                return loaded
        return None

    def to_dict(self) -> dict:
        return {
            "backend_kind": self.backend_kind,
            "reachable": self.reachable,
            "skills": list(self.skills),
            "error": self.error,
        }


def _probe_one_backend(backend_kind: str) -> AgentSkillProbe:
    """Connect a fresh AgentLayer session for ``backend_kind`` and read its skill list.

    Callers test for a specific skill via ``probe.has(...)`` /
    ``probe.resolve(...)``. A backend that declines to answer
    (``None``) is reported the same way a crashed one is: a probe with
    ``reachable=False``, because both mean *we did not learn what is
    installed* — never *nothing is installed*.
    """
    # Imported lazily so module import is cheap.
    from agent_flow.config import AgentLayerConfig, BackendConfig, SessionConfig
    from agent_flow.layers import AgentLayer

    try:
        config = AgentLayerConfig(
            name=f"skill-probe-{backend_kind}",
            backend=BackendConfig(kind=backend_kind, model=_default_model(backend_kind)),
            session=SessionConfig(mode="stateless"),
            print_activity=False,
        )
        with AgentLayer(config) as layer:
            skills = layer.fetch_available_skills()
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as a probe error
        return AgentSkillProbe(
            backend_kind=backend_kind, skills=[], error=f"{type(exc).__name__}: {exc}"
        )
    if skills is None:
        return AgentSkillProbe(
            backend_kind=backend_kind,
            skills=[],
            error="backend reported no skill list",
        )
    return AgentSkillProbe(backend_kind=backend_kind, skills=list(skills))


def _default_model(backend_kind: str) -> str:
    from agent_flow.config import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL

    if backend_kind == "claude-code":
        return CLAUDE_CODE_DEFAULT_MODEL
    if backend_kind == "codex":
        return CODEX_DEFAULT_MODEL
    raise ValueError(f"unknown backend kind: {backend_kind!r}")


def check_skill_via_agent_layer(
    skill: str,
    backend_kinds: Iterable[str] = ("claude-code", "codex"),
) -> dict[str, AgentSkillProbe]:
    """Live-check a skill's availability via real AgentLayer sessions.

    Returns a mapping of ``backend_kind`` -> ``AgentSkillProbe`` so the
    caller can both confirm presence (``probe.has(skill)``) and inspect
    why a probe failed (``probe.error``). Failures don't raise; they
    return a probe with ``reachable=False``.

    One process spawn per backend and no model call, so this is cheap
    enough to sit at import time or on a workflow's launch path.
    """
    # ``skill`` frames the request but isn't needed to probe: each backend
    # returns its full skill list, which callers test via ``probe.has(skill)``.
    del skill
    return {kind: _probe_one_backend(kind) for kind in backend_kinds}


def resolve_first_available_skill(
    candidates: Sequence[str],
    backend_kinds: Iterable[str] = ("claude-code",),
) -> tuple[str | None, bool]:
    """Return ``(loaded_name, probe_ok)`` for the first available candidate.

    ``candidates`` is preference-ordered; the first name any
    reachable backend reports wins, resolved to the spelling the harness
    actually loaded (see :meth:`AgentSkillProbe.resolve`) so a caller can
    quote it back to an agent verbatim.

    ``probe_ok`` is ``False`` when no backend returned a **usable** skill
    list — a transient CLI/SDK failure, not evidence of absence. Callers
    should fail *open* on that (assume the most-preferred candidate and
    let the agent verify in its own session), because silently
    downgrading a stage the user asked for is worse than a wasted check.
    An *empty* list from a reachable backend counts as unusable too: a
    live CLI always names something (its own built-in commands alone), so
    an empty answer is a backend that answered without knowing.
    """
    probes = check_skill_via_agent_layer(candidates[0] if candidates else "", backend_kinds)
    informative = [probe for probe in probes.values() if probe.reachable and probe.skills]
    if not informative:
        return None, False
    for candidate in candidates:
        for probe in informative:
            loaded = probe.resolve(candidate)
            if loaded is not None:
                return loaded, True
    return None, True
