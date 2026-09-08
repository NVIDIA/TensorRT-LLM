"""Deciding and checking what the design agent runs, without running it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.perf_optimize import sweep_design


def _skill(root: Path, where: str = ".agents/skills") -> Path:
    scripts = Path(root) / where / "create-sweep" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / sweep_design.DESIGN_SCRIPT).write_text("# design_sweep\n")
    (scripts / sweep_design.FACTS_SCRIPT).write_text("# model_facts\n")
    return scripts


# ------------------------------------------------------------ locating it


def test_both_skill_directories_are_searched_because_they_disagree(tmp_path):
    """Neither copy is a superset of the other -- this was measured."""
    root_a = tmp_path / "a"
    _skill(root_a, ".agents/skills")
    assert sweep_design.scripts_dir(root_a).is_dir()

    root_b = tmp_path / "b"
    _skill(root_b, ".claude/skills")
    assert sweep_design.scripts_dir(root_b).is_dir()


def test_a_skill_present_but_unreachable_is_named_as_that(tmp_path):
    """A skill on disk but outside the discovered directory.

    "Not installed" and "installed where the agent cannot see it" are
    different problems with the same symptom, and the second one is the
    one that actually happened on this checkout.
    """
    with pytest.raises(sweep_design.SweepDesignError, match="present on disk and uninvokable"):
        sweep_design.scripts_dir(tmp_path / "empty")


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


def test_every_generated_command_names_the_skill_s_own_script(tmp_path):
    scripts = _skill(tmp_path)
    facts = sweep_design.facts_command(scripts, "/ckpt", tmp_path / "facts.json")
    shapes = sweep_design.shapes_command(
        scripts, tmp_path / "facts.json", tmp_path / "s.json", gpu="GB300", per_node=4
    )
    sol = sweep_design.sol_command(scripts, tmp_path / "plan.json", tmp_path / "sol.yaml")
    assert sweep_design.FACTS_SCRIPT in facts
    assert "--gpu-name GB300" in shapes and "--gpus-per-node 4" in shapes
    assert sweep_design.DESIGN_SCRIPT in sol and "--plan" in sol


# ------------------------------------------------------------- the widening


BASE_CTX = {
    "model": {"model_card": "x"},
    "benchmarks": [{"isl": 8192, "osl": 1, "max_batch": [2], "tp_size": [4], "ratio": [0.8]}],
    "gpu_overrides": {"GB300": {"benchmarks": [{"max_batch": [2]}]}},
}


def test_the_ctx_half_gets_candidates_so_it_can_be_selected_not_computed():
    """More candidates is a longer list, not new machinery.

    The sweep format already expands max_batch x tp_size as a product; what
    the widening buys is that the context point comes from measurement.
    """
    out = sweep_design.widen_ctx_candidates(BASE_CTX, batches=[1, 4], tp_sizes=[8])
    row = out["benchmarks"][0]
    assert row["max_batch"] == [1, 2, 4]
    assert row["tp_size"] == [4, 8]


def test_the_sized_answer_is_never_dropped_from_the_candidates():
    """It is the skill's own answer; without it the sweep cannot confirm it."""
    out = sweep_design.widen_ctx_candidates(BASE_CTX, batches=[16], tp_sizes=[8])
    assert 2 in out["benchmarks"][0]["max_batch"]
    assert 4 in out["benchmarks"][0]["tp_size"]


def test_the_override_table_is_dropped_rather_than_merged():
    """Candidates from two places can disagree without anything raising."""
    out = sweep_design.widen_ctx_candidates(BASE_CTX, batches=[1], tp_sizes=[8])
    assert "gpu_overrides" not in out


def test_a_config_with_nothing_to_widen_is_refused(tmp_path):
    with pytest.raises(sweep_design.SweepDesignError, match="silently sweep"):
        sweep_design.widen_ctx_candidates({"model": {}}, batches=[1], tp_sizes=[4])


# --------------------------------------------------------------- the checks


def _plan(entries):
    return {"header": {}, "entries": entries}


def test_a_shape_whose_probe_failed_is_reported_not_silently_dropped():
    """A failed probe leaves its shape out of `entries` rather than failing.

    The sweep that follows would rank a frontier the missing shapes were
    never on, and the report would call it the frontier.
    """
    plan = _plan(
        [
            {"shape": {"kind": "tep", "tp_size": 4, "adp": False}, "max_batch": 64, "mtps": [0]},
            {"shape": {"kind": "dep", "tp_size": 8, "adp": True}, "max_batch": 32, "mtps": [0]},
        ]
    )
    missing = sweep_design.verify_plan(plan, expected_shapes=["tep_4", "dep_8", "dep_16", "dep_32"])
    assert missing == ["dep_16", "dep_32"]


def test_an_empty_plan_is_refused_before_the_sweep_expands_nothing():
    with pytest.raises(sweep_design.SweepDesignError, match="no shape's max batch was"):
        sweep_design.verify_plan(_plan([]), expected_shapes=["tep_4"])


def test_a_guessed_batch_is_refused_because_the_batch_is_measured():
    plan = _plan([{"shape": {"kind": "tep", "tp_size": 4, "adp": False}, "mtps": [0]}])
    with pytest.raises(sweep_design.SweepDesignError, match="measured, never defaulted"):
        sweep_design.verify_plan(plan, expected_shapes=["tep_4"])


SOL_ROW = [1, 1, 4, 64, 256, False, "0.9", 3, 0, "1,32"]


def test_a_token_starved_row_is_refused_before_an_allocation_is_spent():
    """A row whose max_num_tokens is under batch x (mtp+1).

    The generation step is token-starved, so the measured iteration time
    belongs to a smaller batch than the row claims -- and the run still
    succeeds.
    """
    bad = dict(gen_configs=[list(SOL_ROW[:4]) + [128] + list(SOL_ROW[5:])])
    with pytest.raises(sweep_design.SweepDesignError, match=r"max_num_tokens 128 != batch 64"):
        sweep_design.verify_sol_yaml(bad)
    sweep_design.verify_sol_yaml({"gen_configs": [SOL_ROW]})


def test_the_design_sweep_may_not_carry_a_build_source():
    """Fixing the point measures the image.

    A build source here would choose the operating point on code that is not
    what any campaign starts from.
    """
    for key in ("trtllm_install", "trtllm_patch"):
        with pytest.raises(sweep_design.SweepDesignError, match="not what any campaign starts"):
            sweep_design.verify_sol_yaml({"gen_configs": [SOL_ROW], key: {"trtllm_repo": "/r"}})


def test_a_generated_file_that_cannot_be_read_names_itself(tmp_path):
    with pytest.raises(sweep_design.SweepDesignError, match="could not read"):
        sweep_design.load_yaml(tmp_path / "missing.yaml")


def test_a_real_round_trip_through_the_widening(tmp_path):
    """What the agent writes, this module reads back and checks."""
    path = tmp_path / "ctx.yaml"
    path.write_text(yaml.safe_dump(BASE_CTX), encoding="utf-8")
    widened = sweep_design.widen_ctx_candidates(
        sweep_design.load_yaml(path), batches=[1, 4], tp_sizes=[8]
    )
    path.write_text(yaml.safe_dump(widened), encoding="utf-8")
    assert sweep_design.load_yaml(path)["benchmarks"][0]["max_batch"] == [1, 2, 4]
    assert json.dumps(widened)  # serializable: no yaml-only types leaked in


# ---------------------------------------------------------- the instruction


def _instruction(tmp_path):
    return sweep_design.designer_instruction(
        scripts=_skill(tmp_path),
        design_dir=tmp_path / "sweep_design",
        shapes=["tep_4", "dep_8"],
        facts=tmp_path / "facts.json",
        ctx_config=tmp_path / "ctx.yaml",
        sol_yaml=tmp_path / "sol.yaml",
        plan=tmp_path / "plan.json",
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
    assert "corrupted dependency download" in text
    assert "different findings" in text


def test_the_widened_ctx_sweep_may_not_be_narrowed_by_the_agent(tmp_path):
    text = _instruction(tmp_path)
    assert "do not narrow it" in text
