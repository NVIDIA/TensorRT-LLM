"""Guards on the Python-side resolution of the projector's methodology skill.

``internal-perf-sol-analysis`` carries the ``internal-`` prefix, so
open-source builds of the ``trtllm-agent-toolkit`` plugin strip it while
keeping ``perf-analysis``. Which one a session has is resolved here, once
per run, so the projector is told to load a skill that is actually there.

The failure mode pinned here is **failing closed**: a transient probe
failure must never read as "absent" and silently downgrade a stage the
user asked for. The bare/qualified name resolution the probe rests on is
``agent_flow.utils``' own contract and is pinned in ``tests/test_utils.py``.
"""

from __future__ import annotations

from agent_flow.utils import AgentSkillProbe, resolve_first_available_skill
from agent_flow.workflows.perf_analyze.sol_methodology import (
    SOL_SKILL_CANDIDATES,
    SolMethodology,
    output_instruction,
    projector_instruction,
    qualified_skill_name,
    resolve_sol_methodology,
)

_QUALIFIED_SOL = "trtllm-agent-toolkit:internal-perf-sol-analysis"
_QUALIFIED_FALLBACK = "trtllm-agent-toolkit:perf-analysis"


# --------------------------------------------------------------------------- #
# resolve_sol_methodology — the preference order, and an unrunnable probe
# --------------------------------------------------------------------------- #


def _stub_probes(monkeypatch, skills: list[str] | None, *, reachable: bool = True) -> None:
    probe = AgentSkillProbe(
        backend_kind="claude-code",
        skills=skills or [],
        error=None if reachable else "RuntimeError: boom",
    )
    monkeypatch.setattr(
        "agent_flow.utils.check_skill_via_agent_layer",
        lambda skill, backend_kinds=(): {"claude-code": probe},
    )


def test_resolution_prefers_the_internal_skill(monkeypatch):
    _stub_probes(monkeypatch, [_QUALIFIED_SOL, _QUALIFIED_FALLBACK])
    methodology = resolve_sol_methodology()
    assert methodology.name == "full"
    assert methodology.skill == _QUALIFIED_SOL
    assert not methodology.degraded
    # Nothing notable to tell the operator on the happy path.
    assert methodology.console_note() is None


def test_resolution_falls_back_to_perf_analysis(monkeypatch):
    """The open-source-toolkit case: internal skills stripped, the rest kept."""
    _stub_probes(
        monkeypatch, [_QUALIFIED_FALLBACK, "trtllm-agent-toolkit:perf-optimization-casebook"]
    )
    methodology = resolve_sol_methodology()
    assert methodology.name == "reduced"
    assert methodology.skill == _QUALIFIED_FALLBACK
    assert methodology.degraded
    note = methodology.console_note()
    assert note and "not installed" in note and "perf-analysis" in note


def test_resolution_keeps_the_full_brief_when_neither_is_installed(monkeypatch):
    """Nothing to fall back to, so nothing to swap in.

    The full prompt's own last bullet already covers a methodology skill
    that will not load — inventing a third brief for this case would only
    duplicate it.
    """
    _stub_probes(monkeypatch, ["some-unrelated-skill"])
    methodology = resolve_sol_methodology()
    assert methodology.name == "full"
    assert methodology.skill is None
    note = methodology.console_note()
    assert note and "neither" in note


def test_an_unreachable_probe_fails_open_to_the_full_methodology(monkeypatch):
    """Absence of evidence is not evidence of absence.

    A transient CLI/SDK failure must not silently downgrade a stage the
    user asked for — the prompt's own fallback catches a wrong guess.
    """
    _stub_probes(monkeypatch, [], reachable=False)
    methodology = resolve_sol_methodology()
    assert methodology.name == "full"
    assert not methodology.probed
    note = methodology.console_note()
    assert note and "could not read" in note


def test_a_reachable_but_empty_probe_fails_open(monkeypatch):
    """An empty skill list is a failed probe, not an empty environment.

    A live CLI always names something — its own built-in commands alone —
    so ``skills=[] / error=None`` is a backend that answered without
    knowing. Read literally it says "absent", which would degrade a
    fully-equipped machine.
    """
    _stub_probes(monkeypatch, [])
    methodology = resolve_sol_methodology()
    assert methodology.name == "full"
    assert not methodology.probed


def test_a_raising_probe_fails_open_too(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("no backend here")

    monkeypatch.setattr("agent_flow.utils.check_skill_via_agent_layer", _boom)
    methodology = resolve_sol_methodology()
    assert methodology.name == "full"
    assert not methodology.probed


def test_a_disabled_stage_never_probes(monkeypatch):
    """`sol.enabled: false` must not pay for a live session."""

    def _fail(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("probe ran for a disabled stage")

    monkeypatch.setattr("agent_flow.utils.check_skill_via_agent_layer", _fail)
    assert resolve_sol_methodology(enabled=False) == SolMethodology()


def test_the_default_methodology_is_full_and_costs_nothing():
    """Constructing a workflow directly (tests included) never probes."""
    methodology = SolMethodology()
    assert (methodology.name, methodology.probed, methodology.degraded) == ("full", True, False)
    assert methodology.skill == SOL_SKILL_CANDIDATES[0]


def test_candidate_order_is_the_preference_order(monkeypatch):
    assert SOL_SKILL_CANDIDATES == ("internal-perf-sol-analysis", "perf-analysis")
    # Candidate order beats the order the backend happens to list them in.
    _stub_probes(monkeypatch, [_QUALIFIED_FALLBACK, _QUALIFIED_SOL])
    assert resolve_first_available_skill(SOL_SKILL_CANDIDATES) == (_QUALIFIED_SOL, True)


# --------------------------------------------------------------------------- #
# The two driving-message paragraphs that vary with the resolved methodology
# --------------------------------------------------------------------------- #


def test_full_instruction_quotes_the_loaded_name_without_a_hint():
    """A probed run knows the spelling, so the agent never has to guess."""
    message = projector_instruction(SolMethodology(skill=_QUALIFIED_SOL))
    assert f"**load the `{_QUALIFIED_SOL}` skill**" in message
    assert "peaks calculator" in message
    assert "α-β-u" in message
    assert "if the bare name is not found" not in message


def test_an_unresolved_name_still_offers_both_spellings():
    """A bare name is a guess — the workflow never probed, or the probe failed.

    That is the message this stage has always sent, so the guess keeps the
    plugin-qualified alternative rather than becoming a guaranteed miss.
    """
    for methodology in (SolMethodology(), SolMethodology(probed=False)):
        message = projector_instruction(methodology)
        assert "**load the `internal-perf-sol-analysis` skill**" in message
        assert qualified_skill_name("internal-perf-sol-analysis") in message
        assert "if the bare name is not found" in message


def test_the_reduced_instruction_names_perf_analysis_and_never_a_calculator():
    message = projector_instruction(SolMethodology(name="reduced", skill=_QUALIFIED_FALLBACK))
    assert f"**Load the `{_QUALIFIED_FALLBACK}` skill**" in message
    assert "not installed in this session" in message
    # There is no peaks calculator to point at, and no ceiling may be faked.
    assert "peaks calculator you do not have" in message
    assert "unavailable form" in message


def test_the_output_instruction_persists_the_peaks_file_only_with_a_calculator():
    full = output_instruction(
        SolMethodology(), "/ws/sol_projection.md", "/ws/sol_work/peaks.json", "the Analyzer"
    )
    assert "persist the machine-readable peaks file" in full
    assert "/ws/sol_work/peaks.json` — the Analyzer joins against it" in full

    reduced = output_instruction(
        SolMethodology(name="reduced", skill=_QUALIFIED_FALLBACK),
        "/ws/sol_projection.md",
        "/ws/sol_work/peaks.json",
        "the Analyzer",
    )
    # A hand-made peaks file would later be read as calculator output.
    assert "do **not** write `/ws/sol_work/peaks.json`" in reduced
    assert "persist the machine-readable peaks file" not in reduced


def test_every_output_instruction_keeps_the_honest_escape_hatch():
    for methodology in (SolMethodology(), SolMethodology(name="reduced", skill="perf-analysis")):
        message = output_instruction(
            methodology, "/ws/sol_projection.md", "/ws/sol_work/peaks.json", "the Analyzer"
        )
        assert "Projection unavailable: <reason>" in message
        assert "never fabricate numbers" in message
        # Both write the same file with the same section list.
        assert "Projection setup / Projected SOL ceiling / Measured vs SOL" in message


def test_qualified_skill_name_adds_the_plugin_and_is_idempotent():
    assert qualified_skill_name("internal-perf-sol-analysis") == _QUALIFIED_SOL
    assert qualified_skill_name(_QUALIFIED_SOL) == _QUALIFIED_SOL
