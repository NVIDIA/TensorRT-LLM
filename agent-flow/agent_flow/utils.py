"""Check whether a skill is loaded by the Claude and Codex harnesses.

The check is performed by spinning up a real ``AgentLayer`` session for
each backend and reading the skill list off its ``SessionInitEvent``.
This reflects exactly what each harness loaded at session start, which
is the only authoritative answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class AgentSkillProbe:
    """Result of probing one backend via ``AgentLayer.fetch_session_init``."""

    backend_kind: str
    skills: list[str] = field(default_factory=list)
    error: str | None = None  # None on success; str message when the probe failed

    @property
    def reachable(self) -> bool:
        return self.error is None

    def has(self, skill: str) -> bool:
        return skill in self.skills

    def to_dict(self) -> dict:
        return {
            "backend_kind": self.backend_kind,
            "reachable": self.reachable,
            "skills": list(self.skills),
            "error": self.error,
        }


def _probe_one_backend(backend_kind: str) -> AgentSkillProbe:
    """Spin up a fresh AgentLayer session for ``backend_kind`` and read its skill list.

    Reads the full skill list off the ``SessionInitEvent``. Callers test for a specific
    skill via ``probe.has(...)``.
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
            event = layer.fetch_session_init()
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as a probe error
        return AgentSkillProbe(
            backend_kind=backend_kind, skills=[], error=f"{type(exc).__name__}: {exc}"
        )
    return AgentSkillProbe(backend_kind=backend_kind, skills=list(event.skills))


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
    """
    # ``skill`` frames the request but isn't needed to probe: each backend
    # returns its full skill list, which callers test via ``probe.has(skill)``.
    del skill
    return {kind: _probe_one_backend(kind) for kind in backend_kinds}
