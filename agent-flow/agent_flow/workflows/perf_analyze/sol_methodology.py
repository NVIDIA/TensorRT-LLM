"""Resolve which SOL methodology skill the projector stage can actually use.

The projector's methodology is the ``internal-perf-sol-analysis`` skill,
whose ``internal-`` prefix means open-source builds of the
``trtllm-agent-toolkit`` plugin strip it while keeping ``perf-analysis``.
Which of them a session has is settled **here**, once per run, against
the live skill list, so the projector is told to load a skill that is
actually there.

The fallback is deliberately small: ``perf-analysis`` instead of the SOL
skill, no peaks file, and the projector's own prompt already says what
to do without a peaks calculator. Everything downstream — the analyzer's
correlation, the reporter's weighing — keeps the prompts it has always
had and degrades on its own, as it always could.

Shared with perf-optimize, which imports this module directly: it is the
same stage with the same dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

# Preference-ordered: the first that the session actually loaded wins;
# bare names match a plugin-qualified install (``AgentSkillProbe.resolve``).
SOL_SKILL_CANDIDATES = ("internal-perf-sol-analysis", "perf-analysis")

# The full methodology's skill, and what we assume when the probe could
# not run (see ``SolMethodology.probed``).
_SOL_SKILL = "internal-perf-sol-analysis"

# The plugin that ships ``_SOL_SKILL`` — a plugin install is the only way it
# ships, and the harness loads a plugin skill *only* under its
# ``<plugin>:<name>`` spelling. A probed run quotes the loaded name back
# verbatim and never needs this; a fail-open run is guessing, so it has to
# offer both spellings or its guess becomes a guaranteed miss.
_SOL_PLUGIN = "trtllm-agent-toolkit"


def qualified_skill_name(skill: str) -> str:
    """``skill`` under the plugin spelling the harness loads it by."""
    return skill if ":" in skill else f"{_SOL_PLUGIN}:{skill}"


@dataclass(frozen=True)
class SolMethodology:
    """Which SOL methodology the projector stage can run.

    ``name`` is ``full`` (the SOL skill is there) or ``reduced`` (it is
    not, so ``perf-analysis`` stands in). ``skill`` is the name the
    harness actually loaded, so the driving message can quote it verbatim
    instead of making the agent guess between bare and plugin-qualified
    spellings; it is ``None`` when nothing resolved, which keeps the full
    brief and its own missing-skill fallback. ``probed`` is ``False``
    when no backend was reachable and we **failed open** to ``full``,
    which is far better than silently downgrading a stage the user asked
    for; it only changes what the operator is told, since the message
    then quotes the bare name with its qualified spelling either way.
    """

    name: str = "full"
    skill: str | None = _SOL_SKILL
    probed: bool = True

    @property
    def degraded(self) -> bool:
        return self.name != "full"

    def console_note(self) -> str | None:
        """One line for the operator, or ``None`` when nothing is notable."""
        if not self.probed:
            return (
                "projector: could not read this session's skill list — assuming "
                f"`{_SOL_SKILL}` is available (the stage verifies and degrades on its own)"
            )
        if self.name == "reduced":
            return (
                f"projector: `{_SOL_SKILL}` is not installed "
                f"(it ships only in internal toolkit builds) — falling back to `{self.skill}`, "
                "which grounds no peaks calculator, so the ceiling is coarse and no "
                "peaks file is written"
            )
        if self.skill is None:
            return (
                f"projector: neither `{_SOL_SKILL}` nor `perf-analysis` is installed — the "
                "stage will ground what it can and degrade on its own"
            )
        return None


def resolve_sol_methodology(enabled: bool = True) -> SolMethodology:
    """Probe the live skill list and return the methodology the projector has.

    Costs one backend connection and no model call (~1 s), and is
    skipped entirely when the stage is disabled. Never raises: an
    unreachable probe fails open to ``full``.
    """
    if not enabled:
        return SolMethodology()
    # Imported lazily so importing this module stays free for the test suite.
    from agent_flow.utils import resolve_first_available_skill

    try:
        loaded, probe_ok = resolve_first_available_skill(SOL_SKILL_CANDIDATES)
    except Exception:  # noqa: BLE001 - a probe failure must never fail the run
        return SolMethodology(probed=False)
    if not probe_ok:
        return SolMethodology(probed=False)
    if loaded is None:
        # Nothing to fall back to, so keep the full brief: its own last
        # bullet already covers a methodology skill that will not load.
        return SolMethodology(skill=None)
    # ``loaded`` may be plugin-qualified; match on the bare name.
    if loaded == _SOL_SKILL or loaded.endswith(f":{_SOL_SKILL}"):
        return SolMethodology(skill=loaded)
    return SolMethodology(name="reduced", skill=loaded)


def _skill_load_phrase(skill: str) -> str:
    """``(via the `Skill` tool[; fully-qualified <name> ...])`` for ``skill``.

    A probed run quotes the spelling the harness actually loaded, so the
    agent has nothing to guess. A bare name is a guess — the workflow
    never probed, or the probe could not run — so it carries the
    plugin-qualified alternative the way this message always has.
    """
    if ":" in skill:
        return "(via the `Skill` tool)"
    return (
        f"(via the `Skill` tool; fully-qualified `{qualified_skill_name(skill)}` "
        "if the bare name is not found)"
    )


def projector_instruction(methodology: SolMethodology) -> str:
    """The projector's per-turn methodology paragraph.

    The system prompt carries the methodology itself; this is the driving
    message's echo of it, naming the skill this session actually has.
    """
    if methodology.name == "reduced":
        skill = methodology.skill or "perf-analysis"
        return (
            f"**Load the `{skill}` skill** {_skill_load_phrase(skill)}. "
            f"`{_SOL_SKILL}` is **not installed in this session** (it ships only in "
            "internal builds of the toolkit), so work the *Fallback* section of your "
            "system prompt: read the served model's architecture from the checkpoint's "
            "`config.json`, ground each hardware peak from a named source rather than "
            "the peaks calculator you do not have, and derive the "
            "**speed-of-light (SOL) ceiling** for TTFT, TPOT, and throughput at the "
            "measured operating point — marking clearly that the peaks are not "
            "calculator-resolved. If nothing defensible can be grounded, write the "
            "unavailable form."
        )
    resolved = methodology.skill or _SOL_SKILL
    return (
        f"Early on, **load the `{resolved}` skill** {_skill_load_phrase(resolved)} — it is "
        "the methodology every projected number comes from, as your system prompt "
        "directs. Read the served model's architecture from the checkpoint's "
        "`config.json`, resolve the hardware peaks with the skill's peaks calculator "
        "(never from memory), measure the latency constants with its "
        "`measure_channels.py` if a GPU is reachable (record them as unmeasured "
        "otherwise), and derive the **speed-of-light (SOL) ceiling** for TTFT, TPOT, "
        "and throughput at the measured operating point by instantiating the skill's "
        "α-β-u formulas — showing every formula with the numbers substituted."
    )


def output_instruction(
    methodology: SolMethodology,
    projection_path: str,
    peaks_path: str,
    peaks_consumer: str,
) -> str:
    """The projector's per-turn output paragraph.

    The peaks file is the one deliverable the fallback cannot produce:
    ``sol_calc.py`` ships with the missing skill, so nothing reads it and
    a hand-made one would later be mistaken for calculator output.
    """
    sections = (
        "Projection setup / Projected SOL ceiling / Measured vs SOL / Headroom & bound "
        "mix / Guidance for optimization / Caveats"
    )
    if methodology.name == "reduced":
        return (
            f"`Write` your projection to `{projection_path}` using the required "
            f"structure in your system prompt ({sections}), and do **not** write "
            f"`{peaks_path}` — without the peaks calculator there is nothing "
            "schema-honest to put in it. If no defensible ceiling could be grounded, "
            "write the unavailable form (`Projection unavailable: <reason>`) — never "
            "fabricate numbers."
        )
    return (
        f"`Write` your projection to `{projection_path}` using the required structure in "
        f"your system prompt ({sections}), and persist the machine-readable peaks file "
        f"(latency constants merged in) to `{peaks_path}` — {peaks_consumer} joins "
        "against it. If no defensible ceiling could be grounded, write the unavailable "
        "form (`Projection unavailable: <reason>`) — never fabricate numbers."
    )
