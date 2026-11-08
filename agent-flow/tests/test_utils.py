"""Tests for the live skill probe (``agent_flow.utils``).

The probe's contract is narrow and easy to get wrong in one specific
direction: *not learning* what a backend has must never be reported as
*the backend has nothing*. Everything here pins that boundary, plus the
bare/qualified name resolution callers quote back to an agent.
"""

from __future__ import annotations

from unittest.mock import patch

from agent_flow.utils import (
    AgentSkillProbe,
    check_skill_via_agent_layer,
    resolve_first_available_skill,
)

_QUALIFIED = "trtllm-agent-toolkit:internal-perf-sol-analysis"


class _FakeLayer:
    """Stand-in for ``AgentLayer`` as ``_probe_one_backend`` uses it."""

    def __init__(self, result):
        self._result = result

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def fetch_available_skills(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _patch_layer(result):
    return patch("agent_flow.layers.AgentLayer", lambda config: _FakeLayer(result))


# --------------------------------------------------------------------------- #
# check_skill_via_agent_layer — what a probe reports
# --------------------------------------------------------------------------- #


def test_probe_reports_the_skills_the_backend_lists():
    with _patch_layer([_QUALIFIED, "clear"]):
        probes = check_skill_via_agent_layer("whatever", ("claude-code",))

    probe = probes["claude-code"]
    assert probe.reachable
    assert probe.skills == [_QUALIFIED, "clear"]
    assert probe.has(_QUALIFIED)
    assert probe.resolve("internal-perf-sol-analysis") == _QUALIFIED


def test_probe_treats_a_backend_that_cannot_say_as_unreachable():
    """``None`` is "we did not learn", so it must not read as an empty env.

    A probe that returned ``skills=[]`` with ``error=None`` here would
    look exactly like a machine with nothing installed, and callers would
    downgrade a fully-equipped session on the strength of a CLI hiccup.
    """
    with _patch_layer(None):
        probe = check_skill_via_agent_layer("whatever", ("claude-code",))["claude-code"]

    assert not probe.reachable
    assert probe.skills == []
    assert probe.error == "backend reported no skill list"


def test_probe_surfaces_a_backend_failure_instead_of_raising():
    with _patch_layer(RuntimeError("no credentials")):
        probe = check_skill_via_agent_layer("whatever", ("claude-code",))["claude-code"]

    assert not probe.reachable
    assert probe.error == "RuntimeError: no credentials"


def test_probe_covers_every_requested_backend():
    with _patch_layer([_QUALIFIED]):
        probes = check_skill_via_agent_layer("whatever", ("claude-code", "codex"))

    assert set(probes) == {"claude-code", "codex"}


# --------------------------------------------------------------------------- #
# resolve_first_available_skill — the answer a caller quotes to an agent
# --------------------------------------------------------------------------- #


def test_resolve_returns_the_loaded_spelling_of_the_first_candidate():
    with _patch_layer(["trtllm-agent-toolkit:perf-analysis", _QUALIFIED]):
        assert resolve_first_available_skill(("internal-perf-sol-analysis", "perf-analysis")) == (
            _QUALIFIED,
            True,
        )


def test_resolve_falls_open_when_the_probe_could_not_answer():
    with _patch_layer(None):
        assert resolve_first_available_skill(("internal-perf-sol-analysis",)) == (None, False)


def test_resolve_reports_an_answered_miss_as_evidence_of_absence():
    with _patch_layer(["clear", "compact"]):
        assert resolve_first_available_skill(("internal-perf-sol-analysis",)) == (None, True)


def test_resolve_handles_an_empty_candidate_list():
    with _patch_layer(["clear"]):
        assert resolve_first_available_skill(()) == (None, True)


# --------------------------------------------------------------------------- #
# AgentSkillProbe.resolve — bare vs plugin-qualified names
# --------------------------------------------------------------------------- #


def test_resolve_matches_a_plugin_qualified_install_from_a_bare_name():
    probe = AgentSkillProbe(backend_kind="claude-code", skills=[_QUALIFIED])
    # The trap ``resolve`` exists for: exact membership says "absent" for a
    # skill that is plainly installed, because the harness lists only the
    # qualified name.
    assert not probe.has("internal-perf-sol-analysis")
    assert probe.resolve("internal-perf-sol-analysis") == _QUALIFIED


def test_resolve_returns_the_loaded_spelling_verbatim():
    """Callers quote the result back to an agent, so it must be exact."""
    probe = AgentSkillProbe(backend_kind="claude-code", skills=[_QUALIFIED])
    assert probe.resolve(_QUALIFIED) == _QUALIFIED


def test_resolve_prefers_an_exact_match_over_a_plugin_suffix():
    """A local unqualified skill is never shadowed by a plugin's same-named one."""
    probe = AgentSkillProbe(
        backend_kind="claude-code",
        skills=["perf-analysis", "trtllm-agent-toolkit:perf-analysis"],
    )
    assert probe.resolve("perf-analysis") == "perf-analysis"


def test_resolve_does_not_match_across_plugins_or_on_a_partial_name():
    probe = AgentSkillProbe(backend_kind="claude-code", skills=["other:perf-analysis"])
    # A *qualified* request names one plugin and must not match another's.
    assert probe.resolve("trtllm-agent-toolkit:perf-analysis") is None
    # A partial name is not a suffix match either.
    assert probe.resolve("analysis") is None
