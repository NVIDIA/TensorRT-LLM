"""Deciding and checking what the design agent runs, without running it."""

from __future__ import annotations

from agent_flow.workflows.perf_optimize import sweep_design

# ------------------------------------------------------------ the commands


def test_the_scored_command_is_the_anchor_free_one(tmp_path):
    """The skill ends Phase 3 with the join; the staged scope has none.

    Generated rather than instructed, because "use the other post-processor"
    is the kind of instruction honoured on the first run and forgotten on the
    second -- and the failure is a frontier that looks like a frontier.
    """
    command = sweep_design.postprocess_command(tmp_path / "run")
    assert "get_gen_only_perf" in command
    assert "process frontier" not in command
    assert "ctx_json" not in command


# ---------------------------------------------------------- the instruction


def _instruction(tmp_path):
    return sweep_design.designer_instruction(
        model_dir="deepseek-V4-Pro",
        design_dir=tmp_path / "sweep_design",
        tracks=["ctx", "gen"],
    )


def test_the_scored_command_is_handed_over_verbatim_not_described(tmp_path):
    """A described command is one an agent can restate differently later.

    The session this was written after did exactly that: the prompt stated
    the scope in prose and the agent ran the joined post-processor anyway.
    """
    text = _instruction(tmp_path)
    assert "get_gen_only_perf" in text
    assert "Do **not** run `ibc-bench process frontier`" in text
    assert "--ctx_json" in text  # ...named, in the prohibition


def test_the_reason_the_forbidden_command_is_dangerous_is_given(tmp_path):
    """It would still emit a curve, and the curve would still look correct."""
    text = _instruction(tmp_path)
    assert "still look" in text and "correct" in text


def test_the_agent_is_told_not_to_select_the_point(tmp_path):
    """Selection needs a stated preference the design agent was never given."""
    text = _instruction(tmp_path)
    assert "Do not select the operating point" in text


def test_a_failed_shape_must_be_reported_not_omitted(tmp_path):
    text = _instruction(tmp_path)
    assert "never omitted" in text
    assert "a frontier it was never on" in text


def test_infrastructure_failure_and_a_memory_wall_are_kept_apart(tmp_path):
    """One is retried, the other is a finding -- conflating them loses a shape."""
    text = _instruction(tmp_path)
    assert "never started" in text
    assert "different findings" in text


def test_the_skill_s_own_output_placement_rule_is_deferred_to(tmp_path):
    """The rule an earlier generate-the-commands version had no place for.

    `SKILL.md` writes generated YAMLs into `sweep_design/` for an existing
    model dir, never on top of the curated ones. Wrapping its scripts took
    the output path as a free parameter with no guard, trading a
    recomputable mistake for an unrecoverable one.
    """
    text = _instruction(tmp_path)
    assert "never on" in text and "curated" in text
    assert "sweep_design/" in text


def test_the_agent_is_told_to_invoke_the_skill_not_handed_its_commands(tmp_path):
    """The agent invokes the skill rather than being handed its commands.

    The phases are not four commands: they are also gates, a resume spine
    and an output-placement rule.
    """
    text = _instruction(tmp_path)
    assert "invoke the **create-sweep** skill and" in text
    assert "design_sweep.py" not in text
    assert "model_facts.py" not in text


def test_the_irreversible_phases_are_excluded_with_the_reason(tmp_path):
    text = _instruction(tmp_path)
    assert "irreversible" in text
    assert "accept rate" in text
